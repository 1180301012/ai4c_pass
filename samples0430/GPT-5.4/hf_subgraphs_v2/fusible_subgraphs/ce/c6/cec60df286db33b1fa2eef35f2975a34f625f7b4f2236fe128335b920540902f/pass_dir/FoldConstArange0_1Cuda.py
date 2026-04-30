import torch
import triton
import triton.language as tl
from torch import device
from pass_dir.shared_routes import replacement_func


# Match the exact constant arange op from model.py.
def pattern():
    tmp_0 = torch.arange(0, 1, device=device(type='cuda', index=0))
    return tmp_0


# Shared route dispatch; no tensor arguments needed.
def replacement_args():
    return ("arange_0_1_cuda",)


@triton.jit
def _const_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < 1
    zeros = tl.zeros((BLOCK_SIZE,), dtype=tl.int64)
    tl.store(out_ptr + offsets, zeros, mask=mask)