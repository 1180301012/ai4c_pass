import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _arange_1024_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    tl.store(out_ptr + offsets, offsets.to(tl.int64))


@triton.jit
def _cast_bool_1024_kernel(in_ptr, out_ptr, total_n, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_n
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    result = (x != 0).to(tl.int8)
    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def _fused_arange_cast_bool_1024(in_0):
    # Arange: produce [0, 1, 2, ..., 1023] as int64
    arange_out = torch.empty(1024, dtype=torch.int64, device=in_0.device)
    _arange_1024_kernel[(1,)](arange_out, BLOCK_SIZE=1024)

    # Cast in_0 (int64) to bool
    total_n = in_0.numel()
    cast_out = torch.empty_like(in_0, dtype=torch.bool)
    BLOCK_SIZE_CAST = 1024
    grid_cast = ((total_n + BLOCK_SIZE_CAST - 1) // BLOCK_SIZE_CAST,)
    _cast_bool_1024_kernel[grid_cast](in_0, cast_out, total_n, BLOCK_SIZE=BLOCK_SIZE_CAST)

    return (arange_out, cast_out)


def pattern(in_0):
    tmp_1 = torch.arange(0, 1024, device=device(type='cuda', index=0))
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return (tmp_1, tmp_2)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_arange_cast_bool_1024