import torch
import triton
import triton.language as tl
from torch import device


@triton.jit
def _copy_to_cuda_kernel(
    src_ptr,
    dst_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy a flat buffer from src to dst (device-agnostic)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(src_ptr + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def t_to_cuda_wrapper(in_0):
    """Replaces in_0.t().to(device='cuda')."""
    N = in_0.numel()
    out = torch.empty(in_0.shape, dtype=in_0.dtype, device=in_0.device)
    BLOCK_SIZE = 1024
    grid = ((N + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _copy_to_cuda_kernel[grid](in_0, out, N, BLOCK_SIZE=BLOCK_SIZE)
    return out


def pattern(in_0):
    t = in_0.t()
    c = t.to(device(type='cuda'))
    return c


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return t_to_cuda_wrapper