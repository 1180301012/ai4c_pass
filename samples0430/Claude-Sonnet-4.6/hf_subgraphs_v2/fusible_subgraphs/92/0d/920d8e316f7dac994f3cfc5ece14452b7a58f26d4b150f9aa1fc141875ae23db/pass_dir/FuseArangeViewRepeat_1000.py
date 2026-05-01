import torch
from torch import device
import triton
import triton.language as tl


def pattern():
    tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    return ()


@triton.jit
def _arange_repeat_kernel_1000(
    out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    values = offsets.to(tl.int64)
    # Write row 0: out[0, j] = j
    tl.store(out_ptr + offsets, values, mask=mask)
    # Write row 1: out[1, j] = j  (starts at offset N)
    tl.store(out_ptr + N + offsets, values, mask=mask)


@torch.fx.wrap
def arange_repeat_1000():
    N = 1000
    out = torch.empty((2, N), dtype=torch.int64, device='cuda')
    BLOCK_SIZE = 1024
    grid = (1,)
    _arange_repeat_kernel_1000[grid](out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return arange_repeat_1000