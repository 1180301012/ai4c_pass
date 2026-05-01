import torch
import triton
import triton.language as tl

# Pattern matching
def pattern(in_0, in_1):
    tmp_0 = in_0[1]
    tmp_1 = in_0[0]
    tmp_2 = in_1.index_select(-2, tmp_1)
    return tmp_0, tmp_2

# Argument extraction
def replacement_args(in_0, in_1):
    return (in_1, in_0[0])

# Triton kernel
@triton.jit
def index_select_kernel(
    x_ptr,
    indices_ptr,
    out_ptr,
    N,  # size of x[0]
    D,  # size of x[1]
    M,  # size of indices
    BLOCK_M: tl.constexpr,
):
    start_idx = tl.program_id(0) * BLOCK_M
    indices_ptr += start_idx
    out_ptr += start_idx * D

    for i in range(BLOCK_M):
        idx = tl.load(indices_ptr + i)
        x_row = tl.load(x_ptr + idx * D, mask=idx < N, other=0.0)
        tl.store(out_ptr + i * D, x_row)

# Kernel wrapper
@torch.fx.wrap
def optimized_index_select(x, indices):
    N = x.shape[0]
    D = x.shape[1]
    M = indices.shape[0]
    out = torch.empty((M, D), dtype=x.dtype, device=x.device)
    BLOCK_M = 64
    grid = (triton.cdiv(M, BLOCK_M),)
    index_select_kernel[grid](
        x_ptr=x,
        indices_ptr=indices,
        out_ptr=out,
        N=N,
        D=D,
        M=M,
        BLOCK_M=BLOCK_M,
    )
    return out

def replacement_func():
    return optimized_index_select