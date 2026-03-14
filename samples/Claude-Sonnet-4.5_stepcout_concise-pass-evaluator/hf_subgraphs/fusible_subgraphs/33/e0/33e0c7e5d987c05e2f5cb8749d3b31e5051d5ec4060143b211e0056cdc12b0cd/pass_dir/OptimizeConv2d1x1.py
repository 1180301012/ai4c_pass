import torch
import triton
import triton.language as tl


def pattern(input_tensor, weight_tensor, bias_tensor):
    """
    Match 1x1 conv2d with bias
    Must use positional arguments to match the exact call signature in model.py
    """
    result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, (1, 1), (0, 0), (1, 1), 1)
    return result


def replacement_args(input_tensor, weight_tensor, bias_tensor):
    return (input_tensor, weight_tensor, bias_tensor)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_1x1_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ic, stride_ih, stride_iw,
    stride_wn, stride_wc, 
    stride_om, stride_oc, stride_oh, stride_ow,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    """
    Optimized 1x1 conv2d kernel
    Treats 1x1 conv as matrix multiplication: (B*H*W, C_in) x (C_out, C_in).T + bias
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Compute offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        
        # Load input: shape (BLOCK_SIZE_M, BLOCK_SIZE_K)
        input_ptrs = input_ptr + offs_m[:, None] * K + k_offs[None, :]
        input_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        input_block = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight: shape (BLOCK_SIZE_N, BLOCK_SIZE_K)
        # Weight is (C_out, C_in, 1, 1), we treat it as (C_out, C_in)
        weight_ptrs = weight_ptr + offs_n[:, None] * K + k_offs[None, :]
        weight_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        weight_block = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_block, tl.trans(weight_block))
    
    # Load bias
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_block = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    
    # Add bias
    acc += bias_block[None, :]
    
    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)


@torch.fx.wrap
def optimized_conv2d_1x1(input_tensor, weight_tensor, bias_tensor):
    """
    Optimized 1x1 convolution implementation
    """
    # Get shapes
    B, C_in, H, W = input_tensor.shape
    C_out = weight_tensor.shape[0]
    
    # Reshape input: (B, C_in, H, W) -> (B*H*W, C_in)
    input_reshaped = input_tensor.permute(0, 2, 3, 1).contiguous().view(-1, C_in)
    
    # Reshape weight: (C_out, C_in, 1, 1) -> (C_out, C_in)
    weight_reshaped = weight_tensor.squeeze(-1).squeeze(-1).contiguous()
    
    # Prepare for matrix multiplication
    M = B * H * W  # number of spatial positions
    N = C_out      # number of output channels
    K = C_in       # number of input channels
    
    # Allocate output
    output_reshaped = torch.empty((M, N), device=input_tensor.device, dtype=input_tensor.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_SIZE_M']),
        triton.cdiv(N, META['BLOCK_SIZE_N']),
    )
    
    conv2d_1x1_kernel[grid](
        input_reshaped, weight_reshaped, bias_tensor, output_reshaped,
        M, N, K,
        input_reshaped.stride(0), input_reshaped.stride(1), 0, 0,
        weight_reshaped.stride(0), weight_reshaped.stride(1),
        output_reshaped.stride(0), output_reshaped.stride(1), 0, 0,
    )
    
    # Reshape output back: (B*H*W, C_out) -> (B, C_out, H, W)
    output = output_reshaped.view(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()
    
    return output


def replacement_func():
    return optimized_conv2d_1x1