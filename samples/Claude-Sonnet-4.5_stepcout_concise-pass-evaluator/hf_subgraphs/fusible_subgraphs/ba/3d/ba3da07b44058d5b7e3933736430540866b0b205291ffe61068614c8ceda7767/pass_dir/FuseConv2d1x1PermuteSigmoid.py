import torch
import triton
import triton.language as tl


def pattern(bias, weight, input_tensor):
    """
    Pattern: Just Conv2d(1x1) + Permute
    Test to see if basic pattern matching works
    """
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    permuted = conv_out.permute(0, 2, 3, 1)
    return permuted


def replacement_args(bias, weight, input_tensor):
    return (bias, weight, input_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 64, 'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32, 'BLOCK_SIZE_N': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 128, 'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_permute_sigmoid_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_input_batch, stride_input_channel, stride_input_h, stride_input_w,
    stride_weight_out, stride_weight_in,
    stride_output_batch, stride_output_spatial, stride_output_channel,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Fused kernel for 1x1 Conv2d + Permute + Sigmoid
    M: batch_size * H * W (spatial elements)
    N: output_channels
    K: input_channels
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Loop over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        k_mask = k_offs < K
        
        # Load input: need to map linear index to (batch, channel, h, w)
        # offs_m represents flattened (batch, h, w)
        input_ptrs = input_ptr + offs_m[:, None] * stride_input_h + k_offs[None, :] * stride_input_channel
        input_mask = (offs_m[:, None] < M) & k_mask[None, :]
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight: [out_channels, in_channels, 1, 1]
        weight_ptrs = weight_ptr + offs_n[None, :] * stride_weight_out + k_offs[:, None] * stride_weight_in
        weight_mask = (offs_n[None, :] < N) & k_mask[:, None]
        weight_block = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_block, weight_block)
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_block = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias_block[None, :]
    
    # Apply sigmoid
    acc = tl.sigmoid(acc)
    
    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_output_spatial + offs_n[None, :] * stride_output_channel
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)


@torch.fx.wrap
def fused_conv1x1_permute_sigmoid(bias, weight, input_tensor):
    """
    Wrapper function for the fused kernel
    """
    # Input shape: [batch, in_channels, H, W]
    batch, in_channels, H, W = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Reshape input to [batch * H * W, in_channels] conceptually
    # But we'll handle indexing in the kernel
    M = batch * H * W
    N = out_channels
    K = in_channels
    
    # Prepare output: [batch, H, W, out_channels] (after permute, before reshape)
    output = torch.empty((batch, H, W, out_channels), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    # We'll compute flattened but then reshape the output
    output_flat = torch.empty((batch * H * W, out_channels), device=input_tensor.device, dtype=input_tensor.dtype)
    
    fused_conv1x1_permute_sigmoid_kernel[grid](
        input_tensor, weight, bias, output_flat,
        M, N, K,
        input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2), input_tensor.stride(3),
        weight.stride(0), weight.stride(1),
        output_flat.stride(0), 1, output_flat.stride(1),
    )
    
    # Reshape to [batch, H, W, out_channels]
    output = output_flat.reshape(batch, H, W, out_channels)
    
    return output


def replacement_func():
    return fused_conv1x1_permute_sigmoid