import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    """
    Pattern to match Conv2D + LayerNorm + ReLU fusion
    in_0: conv bias
    in_1: conv weight
    in_2: layernorm bias
    in_3: layernorm weight
    in_4: input tensor
    """
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.conv2d(in_4, tmp_1, tmp_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_5 = torch.nn.functional.layer_norm(tmp_4, [tmp_4.shape[1], 1, 1], tmp_3, tmp_2, 1e-05)
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=True)
    return (tmp_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_1, in_0, in_3, in_2)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['N', 'C_out', 'C_in'],
)
@triton.jit
def fused_conv1x1_layernorm_relu_kernel(
    input_ptr, weight_ptr, conv_bias_ptr,
    ln_weight_ptr, ln_bias_ptr, output_ptr,
    N, C_out, C_in,
    stride_in_n, stride_in_c,
    stride_w_cout, stride_w_cin,
    stride_out_n, stride_out_c,
    eps: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for 1x1 Conv2D + LayerNorm + ReLU
    Input: (N, C_in, 1, 1)
    Weight: (C_out, C_in, 1, 1)
    Output: (N, C_out, 1, 1)
    """
    pid_m = tl.program_id(0)  # batch dimension
    pid_n = tl.program_id(1)  # output channel dimension
    
    # Offsets for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    
    # Masks
    mask_m = offs_m < N
    mask_n = offs_n < C_out
    
    # Compute Conv2D output for this block
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    for k_start in range(0, C_in, BLOCK_SIZE_K):
        offs_k = k_start + tl.arange(0, BLOCK_SIZE_K)
        mask_k = offs_k < C_in
        
        # Load input: (BLOCK_SIZE_M, BLOCK_SIZE_K)
        input_ptrs = input_ptr + offs_m[:, None] * stride_in_n + offs_k[None, :] * stride_in_c
        input_block = tl.load(input_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        
        # Load weight: (BLOCK_SIZE_N, BLOCK_SIZE_K)
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_w_cout + offs_k[None, :] * stride_w_cin
        weight_block = tl.load(weight_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        # Accumulate: (BLOCK_SIZE_M, BLOCK_SIZE_N)
        acc += tl.dot(input_block, tl.trans(weight_block))
    
    # Add conv bias
    conv_bias = tl.load(conv_bias_ptr + offs_n, mask=mask_n, other=0.0)
    conv_out = acc + conv_bias[None, :]
    
    # LayerNorm: normalize across C_out dimension (per sample)
    # We need to reduce across all output channels for each sample
    # This requires communication across different blocks handling the same sample
    
    # For simplicity in this block-based approach, we'll compute partial statistics
    # and handle the full layernorm in a separate pass or use atomic operations
    
    # Actually, for layernorm across channels, we need to see all channels per sample
    # Let's use a different approach: process one sample at a time with all channels
    
    # Store conv output temporarily (we'll do layernorm + relu in post-processing)
    output_ptrs = output_ptr + offs_m[:, None] * stride_out_n + offs_n[None, :] * stride_out_c
    tl.store(output_ptrs, conv_out, mask=mask_m[:, None] & mask_n[None, :])


@triton.jit
def layernorm_relu_kernel(
    input_ptr, ln_weight_ptr, ln_bias_ptr, output_ptr,
    N, C,
    stride_n, stride_c,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    LayerNorm + ReLU over channel dimension
    Input/Output: (N, C, 1, 1) -> treated as (N, C)
    """
    pid = tl.program_id(0)  # batch index
    
    if pid >= N:
        return
    
    # Compute mean and variance across channels for this sample
    mean = 0.0
    var = 0.0
    
    for c_start in range(0, C, BLOCK_SIZE):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offs_c < C
        
        # Load values
        ptrs = input_ptr + pid * stride_n + offs_c * stride_c
        vals = tl.load(ptrs, mask=mask, other=0.0)
        
        # Accumulate for mean
        mean += tl.sum(vals)
    
    mean = mean / C
    
    # Compute variance
    for c_start in range(0, C, BLOCK_SIZE):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offs_c < C
        
        ptrs = input_ptr + pid * stride_n + offs_c * stride_c
        vals = tl.load(ptrs, mask=mask, other=0.0)
        
        diff = vals - mean
        var += tl.sum(diff * diff)
    
    var = var / C
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Apply normalization and ReLU
    for c_start in range(0, C, BLOCK_SIZE):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offs_c < C
        
        # Load input
        in_ptrs = input_ptr + pid * stride_n + offs_c * stride_c
        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        
        # Load layernorm parameters
        ln_w = tl.load(ln_weight_ptr + offs_c, mask=mask, other=1.0)
        ln_b = tl.load(ln_bias_ptr + offs_c, mask=mask, other=0.0)
        
        # Normalize
        normalized = (vals - mean) * rstd * ln_w + ln_b
        
        # ReLU
        output = tl.maximum(normalized, 0.0)
        
        # Store
        out_ptrs = output_ptr + pid * stride_n + offs_c * stride_c
        tl.store(out_ptrs, output, mask=mask)


@torch.fx.wrap
def fused_conv1x1_layernorm_relu(input_tensor, weight, conv_bias, ln_weight, ln_bias):
    """
    Fused implementation of 1x1 Conv2D + LayerNorm + ReLU
    
    Args:
        input_tensor: (N, C_in, 1, 1)
        weight: (C_out, C_in, 1, 1)
        conv_bias: (C_out,)
        ln_weight: (C_out, 1, 1)
        ln_bias: (C_out, 1, 1)
    
    Returns:
        output: (N, C_out, 1, 1)
    """
    N, C_in, H, W = input_tensor.shape
    C_out = weight.shape[0]
    
    assert H == 1 and W == 1, "This optimization is for 1x1 spatial dimensions"
    
    # Reshape for easier processing: (N, C, 1, 1) -> (N, C)
    input_2d = input_tensor.squeeze(-1).squeeze(-1)  # (N, C_in)
    ln_weight_1d = ln_weight.squeeze(-1).squeeze(-1)  # (C_out,)
    ln_bias_1d = ln_bias.squeeze(-1).squeeze(-1)  # (C_out,)
    weight_2d = weight.squeeze(-1).squeeze(-1)  # (C_out, C_in)
    
    # Allocate intermediate and output tensors
    conv_out = torch.empty((N, C_out), device=input_tensor.device, dtype=input_tensor.dtype)
    output = torch.empty((N, C_out), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Grid for conv2d
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    grid_conv = (
        triton.cdiv(N, BLOCK_SIZE_M),
        triton.cdiv(C_out, BLOCK_SIZE_N),
    )
    
    # Launch conv kernel
    fused_conv1x1_layernorm_relu_kernel[grid_conv](
        input_2d, weight_2d, conv_bias,
        ln_weight_1d, ln_bias_1d, conv_out,
        N, C_out, C_in,
        input_2d.stride(0), input_2d.stride(1),
        weight_2d.stride(0), weight_2d.stride(1),
        conv_out.stride(0), conv_out.stride(1),
        eps=1e-05,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    
    # Grid for layernorm + relu
    BLOCK_SIZE = 128
    grid_ln = (N,)
    
    # Launch layernorm + relu kernel
    layernorm_relu_kernel[grid_ln](
        conv_out, ln_weight_1d, ln_bias_1d, output,
        N, C_out,
        output.stride(0), output.stride(1),
        eps=1e-05,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape back to (N, C_out, 1, 1)
    output = output.unsqueeze(-1).unsqueeze(-1)
    
    return output


def replacement_func():
    return fused_conv1x1_layernorm_relu