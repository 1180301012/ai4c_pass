import torch
import triton
import triton.language as tl
from torch import device

def pattern(N):
    tmp_0 = torch.arange(0, N, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2

def replacement_args(N):
    return (N,)

@triton.jit
def create_tensor_kernel(
    out_ptr,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 2 * N
    col = offsets % N
    val = col
    tl.store(out_ptr + offsets, val, mask=mask)

@torch.fx.wrap
def kernel_wrapper(N):
    out = torch.empty((2, N), device='cuda')
    BLOCK_SIZE = 1024
    num_programs = (2 * N + BLOCK_SIZE - 1) // BLOCK_SIZE
    create_tensor_kernel[(num_programs,)](
        out_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return out

def replacement_func():
    return kernel_wrapper