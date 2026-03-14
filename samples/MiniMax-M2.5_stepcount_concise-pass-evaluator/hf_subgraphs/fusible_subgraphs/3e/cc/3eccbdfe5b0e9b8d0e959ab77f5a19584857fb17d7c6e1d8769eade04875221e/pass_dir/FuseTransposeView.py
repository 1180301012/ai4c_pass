import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 2}, num_stages=1, num_warps=1),
    ],
    key=[],
)
@triton.jit
def transpose_kernel(
    input_ptr,
    output_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized transpose kernel for small matrices"""
    pid = tl.program_id(0)
    
    # Each program handles one row
    row = pid
    if row >= M:
        return
    
    # Load row and store to column
    for col in range(0, N):
        val = tl.load(input_ptr + row * N + col)
        tl.store(output_ptr + col * M + row, val)


@torch.fx.wrap
def optimized_transpose(x):
    """Optimized transpose for small matrices"""
    M = x.shape[0]
    N = x.shape[1]
    
    out = torch.empty((N, M), dtype=x.dtype, device=x.device)
    
    grid = (M,)
    transpose_kernel[grid](x, out, M, N)
    
    return out


def pattern(x):
    """Match transpose operation"""
    return x.T


def replacement_args(x):
    return (x,)


def replacement_func():
    return optimized_transpose