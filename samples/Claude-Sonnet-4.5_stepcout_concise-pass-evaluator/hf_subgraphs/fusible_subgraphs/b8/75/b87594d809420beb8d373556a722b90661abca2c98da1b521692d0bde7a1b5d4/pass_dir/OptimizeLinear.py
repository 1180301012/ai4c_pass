import torch
import triton
import triton.language as tl

# Pattern matching function for linear layer
def pattern(x, weight, bias):
    """
    Match linear layer: y = x @ weight.T + bias
    """
    result = torch.nn.functional.linear(x, weight, bias)
    return result

def replacement_args(x, weight, bias):
    return (x, weight, bias)

# Optimized Triton kernel for linear layer
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Offsets
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    
    # Compute matrix multiplication: x @ weight.T
    for k in range(0, K, BLOCK_SIZE_K):
        k_mask = (k + offs_k) < K
        
        # Load x: [M, K]
        x_offsets = offs_m[:, None] * stride_xm + (k + offs_k[None, :]) * stride_xk
        x_mask = (offs_m[:, None] < M) & k_mask[None, :]
        x = tl.load(x_ptr + x_offsets, mask=x_mask, other=0.0)
        
        # Load weight: [N, K] (already transposed in memory layout)
        w_offsets = offs_n[:, None] * stride_wn + (k + offs_k[None, :]) * stride_wk
        w_mask = (offs_n[:, None] < N) & k_mask[None, :]
        w = tl.load(weight_ptr + w_offsets, mask=w_mask, other=0.0)
        
        # Accumulate
        acc += tl.dot(x, tl.trans(w))
    
    # Add bias
    bias_offsets = offs_n
    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + bias_offsets, mask=bias_mask, other=0.0)
    acc += bias[None, :]
    
    # Store result
    out_offsets = offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + out_offsets, acc, mask=out_mask)

@torch.fx.wrap
def optimized_linear(x, weight, bias):
    """
    Optimized linear layer using Triton
    y = x @ weight.T + bias
    """
    M = x.shape[0]
    K = x.shape[1]
    N = weight.shape[0]
    
    # Allocate output
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N'])
    )
    
    linear_kernel[grid](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        M=M, N=N, K=K,
        stride_xm=x.stride(0),
        stride_xk=x.stride(1),
        stride_wn=weight.stride(0),
        stride_wk=weight.stride(1),
    )
    
    return out

def replacement_func():
    return optimized_linear