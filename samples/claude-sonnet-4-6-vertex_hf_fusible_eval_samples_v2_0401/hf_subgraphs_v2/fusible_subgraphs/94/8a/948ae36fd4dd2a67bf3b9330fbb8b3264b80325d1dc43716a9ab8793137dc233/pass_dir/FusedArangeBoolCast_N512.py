import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _fused_arange_cast_bool_kernel_512(
    in_ptr,
    bool_ptr,
    arange_ptr,
    bn_elements,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    bool_mask = offsets < bn_elements
    x = tl.load(in_ptr + offsets, mask=bool_mask, other=0)
    bool_val = (x != 0)
    tl.store(bool_ptr + offsets, bool_val, mask=bool_mask)

    arange_mask = offsets < n_elements
    tl.store(arange_ptr + offsets, offsets.to(tl.int64), mask=arange_mask)


_BLOCK_SIZE_512 = 2048


@torch.fx.wrap
def triton_fused_arange_cast_512(in_0):
    n = 512
    bn = in_0.numel()
    arange_out = torch.empty(n, dtype=torch.int64, device=in_0.device)
    bool_out = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    grid = ((bn + _BLOCK_SIZE_512 - 1) // _BLOCK_SIZE_512,)
    _fused_arange_cast_bool_kernel_512[grid](
        in_0, bool_out, arange_out, bn, n, BLOCK_SIZE=_BLOCK_SIZE_512
    )
    return (arange_out, bool_out)


def pattern(in_0):
    tmp_1 = torch.arange(0, 512, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_fused_arange_cast_512