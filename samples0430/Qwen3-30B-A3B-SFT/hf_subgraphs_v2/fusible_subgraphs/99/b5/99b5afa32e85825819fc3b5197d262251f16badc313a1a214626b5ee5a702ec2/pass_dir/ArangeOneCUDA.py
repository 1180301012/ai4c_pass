import torch
from torch import device
import triton
import triton.language as tl


# ── Pattern ──────────────────────────────────────────────────────────────────
def pattern():
    tmp_0 = torch.arange(1, device=device(type='cuda', index=0))
    lazy_load_decompositions = torch._functorch.vmap.lazy_load_decompositions()
    return (tmp_0,)


# ── Argument extractor (none – forward() has no tensor inputs) ───────────────
def replacement_args():
    return ()


# ── Triton kernel: write arange(0, n) as int64 ───────────────────────────────
@triton.jit
def _arange_n_kernel(
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    tl.store(out_ptr + offsets, offsets.to(tl.int64), mask=mask)


# ── Kernel wrapper (no torch.* calls beyond the allocation API) ───────────────
@torch.fx.wrap
def _arange_one_cuda():
    n = 1
    out = torch.empty(n, dtype=torch.int64, device='cuda')
    BLOCK_SIZE = 1
    _arange_n_kernel[(1,)](out, n, BLOCK_SIZE=BLOCK_SIZE)
    return (out,)


# ── Replacement factory (returns the callable, does NOT call it) ──────────────
def replacement_func():
    return _arange_one_cuda