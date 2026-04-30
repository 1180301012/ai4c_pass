import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0):
    tmp_1 = torch.arange(0, 128, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0, "128")


@triton.jit
def fused_arange_tobool_kernel(
    in_ptr, arange_ptr, bool_ptr,
    N, total_elems,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Arange output: write sequential integers for first N elements
    arange_mask = offsets < N
    tl.store(arange_ptr + offsets, offsets.to(tl.int64), mask=arange_mask)

    # Bool conversion: read int64, write bool (non-zero -> True)
    bool_mask = offsets < total_elems
    vals = tl.load(in_ptr + offsets, mask=bool_mask, other=0)
    bool_out = (vals != 0)
    tl.store(bool_ptr + offsets, bool_out, mask=bool_mask)


@torch.fx.wrap
def dispatch_fused_op(in_0, route):
    if route == "128":
        N = 128
    elif route == "256":
        N = 256
    elif route == "512":
        N = 512
    elif route == "1024":
        N = 1024
    else:
        N = 128

    total_elems = in_0.numel()
    arange_out = torch.empty(N, dtype=torch.int64, device=in_0.device)
    bool_out = torch.empty(in_0.shape[0], in_0.shape[1], dtype=torch.bool, device=in_0.device)

    BLOCK_SIZE = 1024
    grid_size = (max(N, total_elems) + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_arange_tobool_kernel[(grid_size,)](
        in_0, arange_out, bool_out,
        N, total_elems,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (arange_out, bool_out)


def replacement_func():
    return dispatch_fused_op