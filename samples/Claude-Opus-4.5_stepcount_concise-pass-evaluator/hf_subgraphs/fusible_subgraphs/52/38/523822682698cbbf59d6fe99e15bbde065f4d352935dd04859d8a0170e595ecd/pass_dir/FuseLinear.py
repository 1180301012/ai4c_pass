import torch
import triton
import triton.language as tl


def pattern(x, weight, bias):
    """
    Match linear layer pattern.
    linear(x, weight, bias)
    """
    out = torch.nn.functional.linear(x, weight, bias)
    return out


def replacement_args(x, weight, bias):
    """Extract arguments needed for the replacement function."""
    return (x, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    # Pointers to matrices
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Linear layer kernel: out = x @ weight.T + bias
    x: [M, K], weight: [N, K], bias: [N], out: [M, N]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # Create block pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over K dimension
    for k in range(0, K, BLOCK_SIZE_K):
        k_offs = k + offs_k
        
        # Load x block [BLOCK_SIZE_M, BLOCK_SIZE_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Load weight block [BLOCK_SIZE_N, BLOCK_SIZE_K] -> transpose to [BLOCK_SIZE_K, BLOCK_SIZE_N]
        w_ptrs = weight_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        # Compute matmul: x @ w.T
        acc += tl.dot(x, tl.trans(w))
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


@torch.fx.wrap
def linear_optimized(x, weight, bias):
    """
    Optimized linear layer using Triton kernel.
    out = x @ weight.T + bias
    """
    M, K = x.shape
    N = weight.shape[0]
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_SIZE_M']),
        triton.cdiv(N, meta['BLOCK_SIZE_N']),
    )
    
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


def replacement_func():
    return linear_optimized