import torch
import triton
import triton.language as tl

BLOCK_SIZE = 256

@triton.jit
def dist_kernel(
    out_ptr,
    N: tl.constexpr,
    grid_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    for idx in range(start_idx, min(start_idx + BLOCK_SIZE, N * N)):
        i = idx // N
        j = idx % N
        i_row = i // grid_size
        j_row = j // grid_size
        diff = i_row - j_row
        squared = diff * diff
        tl.store(out_ptr + idx, 2 * squared)

@torch.fx.wrap
def grid_dist_kernel(dtype, N, grid_size):
    out = torch.empty(N, N, device='cuda', dtype=dtype)
    num_programs = (N * N + BLOCK_SIZE - 1) // BLOCK_SIZE
    dist_kernel[(num_programs,)](
        out_ptr=out,
        N=N,
        grid_size=grid_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out

def pattern(a, b, c):
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim=0)
    tmp_11 = tmp_10.repeat_interleave(14, dim=1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    return tmp_14

def replacement_args(a, b, c):
    return (c.dtype, 196, 14)

def replacement_func():
    return grid_dist_kernel