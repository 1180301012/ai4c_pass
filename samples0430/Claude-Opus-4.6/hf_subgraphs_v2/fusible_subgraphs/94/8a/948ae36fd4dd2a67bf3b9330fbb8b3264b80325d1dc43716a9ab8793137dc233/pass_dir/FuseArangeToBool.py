import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def int64_to_bool_kernel(
    in_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    vals = tl.load(in_ptr + offsets, mask=mask, other=0)
    bool_out = (vals != 0)
    tl.store(out_ptr + offsets, bool_out, mask=mask)


@torch.fx.wrap
def tobool_triton(in_0):
    n_elements = in_0.numel()
    out = torch.empty_like(in_0, dtype=torch.bool)

    BLOCK_SIZE = 2048
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    int64_to_bool_kernel[(grid_size,)](
        in_0, out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return tobool_triton