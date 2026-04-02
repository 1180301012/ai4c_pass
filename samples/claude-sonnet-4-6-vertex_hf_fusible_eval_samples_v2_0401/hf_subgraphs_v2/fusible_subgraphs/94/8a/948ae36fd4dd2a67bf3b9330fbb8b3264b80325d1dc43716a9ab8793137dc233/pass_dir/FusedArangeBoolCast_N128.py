import torch
import triton
import triton.language as tl
from torch import device


# Fused kernel: fills arange_out[i]=i for i<n, and casts in_ptr[i]!=0 to bool_out[i] for i<bn
@triton.jit
def _fused_arange_cast_bool_kernel(
    in_ptr,
    bool_ptr,
    arange_ptr,
    bn_elements,   # total elements for bool cast (B*N)
    n_elements,    # total elements for arange (N)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Bool cast: in_ptr[i] != 0 -> bool_ptr[i]
    bool_mask = offsets < bn_elements
    x = tl.load(in_ptr + offsets, mask=bool_mask, other=0)
    bool_val = (x != 0)
    tl.store(bool_ptr + offsets, bool_val, mask=bool_mask)

    # Arange fill: write i to arange_ptr[i] for i < n_elements
    arange_mask = offsets < n_elements
    tl.store(arange_ptr + offsets, offsets.to(tl.int64), mask=arange_mask)


_BLOCK_SIZE = 2048


@torch.fx.wrap
def triton_fused_arange_cast_128(in_0):
    n = 128
    bn = in_0.numel()
    arange_out = torch.empty(n, dtype=torch.int64, device=in_0.device)
    bool_out = torch.empty(in_0.shape, dtype=torch.bool, device=in_0.device)
    grid = ((bn + _BLOCK_SIZE - 1) // _BLOCK_SIZE,)
    _fused_arange_cast_bool_kernel[grid](
        in_0, bool_out, arange_out, bn, n, BLOCK_SIZE=_BLOCK_SIZE
    )
    return (arange_out, bool_out)


def pattern(in_0):
    tmp_1 = torch.arange(0, 128, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_fused_arange_cast_128