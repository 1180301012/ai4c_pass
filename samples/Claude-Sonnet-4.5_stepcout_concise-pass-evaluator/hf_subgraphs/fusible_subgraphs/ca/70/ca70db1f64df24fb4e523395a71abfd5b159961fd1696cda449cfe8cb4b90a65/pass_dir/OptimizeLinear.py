import torch
import triton
import triton.language as tl

def pattern(input, weight, bias):
    """
    Simple pattern: just optimize linear operation
    """
    output = torch.nn.functional.linear(input, weight, bias)
    return output

def replacement_args(input, weight, bias):
    return (input, weight, bias)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_stages=3, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Optimized linear kernel: output = input @ weight.T + bias
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Matrix multiplication loop
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        # Load input block
        input_ptrs = input_ptr + offs_m[:, None] * stride_im + k_offs[None, :] * stride_ik
        input_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        input_vals = tl.load(input_ptrs, mask=input_mask, other=0.0)
        
        # Load weight block (transposed access)
        weight_ptrs = weight_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        weight_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        weight_vals = tl.load(weight_ptrs, mask=weight_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(input_vals, tl.trans(weight_vals))
    
    # Add bias
    bias_ptrs = bias_ptr + offs_n
    bias_mask = offs_n < N
    bias_vals = tl.load(bias_ptrs, mask=bias_mask, other=0.0)
    acc += bias_vals[None, :]
    
    # Store output
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, acc, mask=output_mask)

@torch.fx.wrap
def optimized_linear(input, weight, bias):
    """
    Optimized linear implementation using Triton
    """
    # Handle batched inputs
    orig_shape = input.shape
    if len(orig_shape) > 2:
        input_2d = input.reshape(-1, orig_shape[-1])
    else:
        input_2d = input
    
    M, K = input_2d.shape
    N, K_w = weight.shape
    
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)
    
    # Grid will be determined by the kernel
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    linear_kernel[grid](
        input_2d, weight, bias, output,
        M, N, K,
        input_2d.stride(0), input_2d.stride(1),
        weight.stride(0), weight.stride(1),
        output.stride(0), output.stride(1),
    )
    
    # Reshape output if input was batched
    if len(orig_shape) > 2:
        output = output.reshape(*orig_shape[:-1], N)
    
    return output

def replacement_func():
    return optimized_linear