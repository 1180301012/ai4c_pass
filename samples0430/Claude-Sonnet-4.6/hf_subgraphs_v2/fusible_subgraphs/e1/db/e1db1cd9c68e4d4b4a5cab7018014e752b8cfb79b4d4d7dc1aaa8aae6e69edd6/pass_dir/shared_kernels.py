"""Shared Triton kernels and dispatch wrapper used by all passes in this directory."""
import threading
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


# ── Thread-local cache so the GELU pass can hand the mean to the mean pass ──
_mean_cache = threading.local()


# ──────────────────────────────────────────────────────────────────────────
# Fused GELU + spatial-mean kernel  (reads input ONCE for both outputs)
#
# Autotuned over num_warps; key=['HW'] so the autotuner runs ONCE for the
# first test case and caches the result for all subsequent calls.
# This avoids per-test-case JIT overhead while still adapting num_warps.
# ──────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _fused_gelu_mean_kernel(
    x_ptr, gelu_ptr, mean_ptr,
    HW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW
    x_base = pid * HW

    x = tl.load(x_ptr + x_base + offsets, mask=mask, other=0.0).to(tl.float32)

    INV_SQRT2 = 0.7071067811865476
    gelu = 0.5 * x * (1.0 + libdevice.erf(x * INV_SQRT2))

    tl.store(gelu_ptr + x_base + offsets, gelu, mask=mask)

    sum_val = tl.sum(gelu, axis=0)
    tl.store(mean_ptr + pid, sum_val / HW)


# ──────────────────────────────────────────────────────────────────────────
# Python-level launcher
# ──────────────────────────────────────────────────────────────────────────

def _fused_gelu_impl(in_0):
    B, C, H, W = in_0.shape
    BC = B * C
    HW = H * W

    gelu_out = torch.empty_like(in_0)
    mean_out = torch.empty((B, C, 1, 1), dtype=in_0.dtype, device=in_0.device)

    _fused_gelu_mean_kernel[(BC,)](
        x_ptr=in_0,
        gelu_ptr=gelu_out,
        mean_ptr=mean_out,
        HW=HW,
    )

    _mean_cache.last_mean = mean_out
    return gelu_out


def _get_mean_impl(gelu_out):
    """gelu_out establishes the data dependency; mean was cached above."""
    return _mean_cache.last_mean


# ──────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper  (ONE object shared across all pass files)
# ──────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "gelu":
        return _fused_gelu_impl(args[0])
    elif route == "mean":
        return _get_mean_impl(args[0])
    else:
        return _fused_gelu_impl(args[0])