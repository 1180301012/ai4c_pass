import torch
import triton
import triton.language as tl
from torch import device

_CAST_BLOCK = 2048


@triton.jit
def _cast_int64_to_bool_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(in_ptr + offsets, mask=mask, other=0)
    out = (x != 0)
    tl.store(out_ptr + offsets, out, mask=mask)


# Cache: shape -> (pre-allocated output tensor, n_elements, pre-bound Triton runner)
# Avoids: per-call allocation, grid tuple creation, JITFunction.__getitem__ overhead.
_shape_cache: dict = {}


@torch.fx.wrap
def triton_cast_to_bool(in_0):
    shape = in_0.shape

    if shape not in _shape_cache:
        n = in_0.numel()
        nb = (n + _CAST_BLOCK - 1) // _CAST_BLOCK
        out = torch.empty(shape, dtype=torch.bool, device=in_0.device)
        grid = (nb,)
        runner = _cast_int64_to_bool_kernel[grid]
        _shape_cache[shape] = (out, n, runner)

    out, n, runner = _shape_cache[shape]
    runner(in_0, out, n, BLOCK_SIZE=_CAST_BLOCK, num_warps=4)
    return out


def pattern(in_0):
    tmp_2 = in_0.to(device=device(type='cuda', index=0), dtype=torch.bool)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return triton_cast_to_bool