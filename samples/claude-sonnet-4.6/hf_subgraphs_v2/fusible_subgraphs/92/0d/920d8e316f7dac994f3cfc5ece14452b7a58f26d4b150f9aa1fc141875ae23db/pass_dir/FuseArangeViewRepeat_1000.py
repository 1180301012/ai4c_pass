import torch
from torch import device
import triton
import triton.language as tl


# ── pattern ──────────────────────────────────────────────────────────────────
def pattern():
    tmp_0 = torch.arange(0, 1000, device=device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return tmp_2


def replacement_args():
    return ()


# ── Triton kernel  ────────────────────────────────────────────────────────────
@triton.jit
def _arange_repeat_1000_kernel(out_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Fused arange(0,1000) + view(1,-1) + repeat(2,1) -> (2,1000) int64 tensor.
    output[flat_idx] = flat_idx % 1000  for flat_idx in [0, 2000)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < 2000            # 2 * 1000
    col = (offsets % 1000).to(tl.int64)
    tl.store(out_ptr + offsets, col, mask=mask)


# ── wrapper ───────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_arange_repeat_1000():
    out = torch.empty((2, 1000), dtype=torch.int64, device='cuda')
    BLOCK_SIZE = 1024
    # ceil(2000 / 1024) = 2 blocks
    _arange_repeat_1000_kernel[(2,)](out, BLOCK_SIZE=BLOCK_SIZE)
    return out


# ── replacement_func ──────────────────────────────────────────────────────────
def replacement_func():
    return fused_arange_repeat_1000