import torch
import triton
import triton.language as tl

# Pattern matching function - matches Conv2D + SiLU + Dropout
def pattern(bias, weight, input_tensor):
    """
    Match the pattern: Conv2D (1x1) -> SiLU -> Dropout(p=0)
    """
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    silu_out = torch.nn.functional.silu(conv_out, inplace=False)
    dropout_out = torch.nn.functional.dropout(silu_out, 0.0, False, False)
    return dropout_out

# Argument extraction function
def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)

# Optimized Triton kernel for fused 1x1 conv + bias + silu
# For 1x1 conv: output[n,c_out,h,w] = sum_c_in(weight[c_out,c_in,0,0] * input[n,c_in,h,w]) + bias[c_out]
# This can be viewed as a batched matrix multiplication over spatial dimensions

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_silu_kernel(
    # Pointers
    input_ptr,      # Input tensor [N, C_in, H, W] viewed as [N*H*W, C_in]
    weight_ptr,     # Weight tensor [C_out, C_in, 1, 1] viewed as [C_out, C_in]
    bias_ptr,       # Bias tensor [C_out]
    output_ptr,     # Output tensor [N, C_out, H, W] viewed as [N*H*W, C_out]
    # Dimensions
    M,              # N * H * W (spatial batch)
    N,              # C_out (output channels)
    K,              # C_in (input channels)
    # Strides for input (viewed as [batch, C_in, H*W])
    stride_in_n,    # stride for batch dimension
    stride_in_c,    # stride for channel dimension
    stride_in_hw,   # stride for spatial dimension
    # Strides for output
    stride_out_n,   # stride for batch dimension
    stride_out_c,   # stride for channel dimension
    stride_out_hw,  # stride for spatial dimension
    # Block sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused 1x1 convolution + bias + SiLU kernel
    Implements: output = SiLU(input @ weight.T + bias)
    Where input is reshaped from [N,C_in,H,W] to [N*H*W, C_in]
    And weight is reshaped from [C_out,C_in,1,1] to [C_out, C_in]
    """
    # Program ID
    pid_m = tl.program_id(0)  # Spatial position (in blocks)
    pid_n = tl.program_id(1)  # Output channel (in blocks)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]
    offs_k = tl.arange(0, BLOCK_K)                     # [BLOCK_K]
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over input channels
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        # Load input block [BLOCK_M, BLOCK_K]
        # Input is in NCHW format, we need to gather from [N*H*W, C_in]
        # For spatial index m, batch = m // (H*W), spatial_pos = m % (H*W)
        # But we can directly use pointer arithmetic with proper strides
        
        # Mask for valid positions
        mask_m = offs_m < M
        mask_k = k_offs < K
        mask_input = mask_m[:, None] & mask_k[None, :]
        
        # Load input - we assume input is NCHW but we iterate over HW first
        # input_ptr points to input[0,0,0,0]
        # For position m in the flattened HW*N dimension and channel c:
        # We need input[batch, c, h, w] where batch*H*W + h*W + w = m
        # Let's use the provided strides
        
        # Actually for 1x1 conv, we can think of input as [N, C_in, H*W]
        # and compute output as [N, C_out, H*W] = weight @ input
        
        # Here M = N_batch * H * W, we iterate over spatial positions
        # For each spatial position m, we want input[m_batch, :, m_hw]
        # But our input is contiguous in NCHW, so we need proper indexing
        
        # Let's compute actual positions
        # m gives us batch*H*W + h*W + w
        # Input shape: [1, 128, 4, 256], so H=4, W=256, HW=1024
        # m / (H*W) = batch, m % (H*W) = spatial position
        
        # Since input is contiguous NCHW, input[n,c,h,w] = input_ptr + n*C*H*W + c*H*W + h*W + w
        # For m = n*H*W + h*W + w and channel c: offset = n*C*H*W + c*H*W + h*W + w
        # = (m // (H*W)) * C*H*W + c*H*W + (m % (H*W))
        # = m + (c-1)*H*W + (batch-1)*(C-1)*H*W... this is getting complex
        
        # Let's use a simpler approach: compute input_offset for each (m, k)
        # where m indexes spatial (across batches), k indexes input channel
        
        # For NCHW layout:
        # input[n, c, h, w] at offset: n * (C_in * H * W) + c * (H * W) + h * W + w
        # We want to access input for spatial index m (= n*H*W + h*W + w) and channel k_offs
        # This requires knowing H*W to extract n and the spatial offset
        
        # With provided strides:
        # stride_in_n = C_in * H * W (batch stride)
        # stride_in_c = H * W (channel stride)
        # stride_in_hw = 1 (spatial stride within a channel plane)
        
        # For spatial position m, we have:
        # n = m // (H*W), spatial = m % (H*W)
        # offset = n * stride_in_n + c * stride_in_c + spatial * stride_in_hw
        # = n * stride_in_n + c * stride_in_c + spatial
        
        # Since stride_in_n = C_in * H * W and spatial = m % (H*W):
        # For the k-th channel: offset = (m // HW) * (C_in * HW) + k * HW + (m % HW)
        
        # To simplify, let HW = stride_in_c (H*W)
        HW = stride_in_c
        
        # Compute batch index and spatial offset for each m
        batch_idx = offs_m // HW
        spatial_idx = offs_m % HW
        
        # Input offsets: [BLOCK_M, BLOCK_K]
        input_offsets = batch_idx[:, None] * stride_in_n + k_offs[None, :] * stride_in_c + spatial_idx[:, None]
        
        a = tl.load(input_ptr + input_offsets, mask=mask_input, other=0.0)
        
        # Load weight block [BLOCK_N, BLOCK_K] and transpose to [BLOCK_K, BLOCK_N]
        # Weight is [C_out, C_in, 1, 1], we want weight[n, k] = weight[n, k, 0, 0]
        mask_n = offs_n < N
        mask_weight = mask_n[:, None] & mask_k[None, :]
        
        # Weight offsets: weight[c_out, c_in, 0, 0] = weight_ptr + c_out * K + c_in
        weight_offsets = offs_n[:, None] * K + k_offs[None, :]  # [BLOCK_N, BLOCK_K]
        b = tl.load(weight_ptr + weight_offsets, mask=mask_weight, other=0.0)
        
        # Matrix multiply: acc += a @ b.T  where a: [BLOCK_M, BLOCK_K], b: [BLOCK_N, BLOCK_K]
        # Result: [BLOCK_M, BLOCK_N]
        acc += tl.dot(a, tl.trans(b))
    
    # Load and add bias
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]
    
    # Apply SiLU: x * sigmoid(x)
    sigmoid_acc = tl.sigmoid(acc)
    result = acc * sigmoid_acc
    
    # Store output
    # Output is [N_batch, C_out, H, W] in NCHW format
    # For spatial position m and output channel n:
    # n_batch = m // (H*W), spatial = m % (H*W)
    # offset = n_batch * stride_out_n + c_out * stride_out_c + spatial
    
    HW = stride_out_c  # H * W
    batch_idx = offs_m // HW
    spatial_idx = offs_m % HW
    
    # Output offsets: [BLOCK_M, BLOCK_N]
    output_offsets = batch_idx[:, None] * stride_out_n + offs_n[None, :] * stride_out_c + spatial_idx[:, None]
    
    mask_m = offs_m < M
    mask_out = mask_m[:, None] & mask_n[None, :]
    
    tl.store(output_ptr + output_offsets, result, mask=mask_out)


@torch.fx.wrap
def fused_conv1x1_silu_dropout(bias, weight, input_tensor):
    """
    Fused 1x1 convolution + SiLU + Dropout(p=0) operation
    Input: [N, C_in, H, W]
    Weight: [C_out, C_in, 1, 1]
    Bias: [C_out]
    Output: [N, C_out, H, W]
    """
    # Get dimensions
    N_batch, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    
    # Create output tensor
    output = torch.empty((N_batch, C_out, H, W), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Compute M (total spatial positions across all batches)
    M = N_batch * H * W
    N = C_out
    K = C_in
    
    # Strides for input NCHW
    stride_in_n = C_in * H * W
    stride_in_c = H * W
    stride_in_hw = 1
    
    # Strides for output NCHW
    stride_out_n = C_out * H * W
    stride_out_c = H * W
    stride_out_hw = 1
    
    # Grid dimensions
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N']),
        )
    
    # Launch kernel
    fused_conv1x1_silu_kernel[grid](
        input_tensor,
        weight,
        bias,
        output,
        M, N, K,
        stride_in_n, stride_in_c, stride_in_hw,
        stride_out_n, stride_out_c, stride_out_hw,
    )
    
    return output


# Replacement function
def replacement_func():
    return fused_conv1x1_silu_dropout