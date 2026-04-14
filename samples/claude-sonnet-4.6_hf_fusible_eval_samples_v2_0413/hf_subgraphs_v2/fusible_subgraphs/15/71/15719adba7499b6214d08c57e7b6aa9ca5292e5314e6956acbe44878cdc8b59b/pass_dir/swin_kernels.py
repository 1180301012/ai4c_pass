"""
Shared Triton kernels and dispatch wrapper for Swin patch-embed fusion passes.
Imported by FuseSwinPatchEmbedTiny.py and FuseSwinPatchEmbedLarge.py.

Strategy: one Triton program per token.
  - Maximum parallelism (N programs) → best latency hiding
  - For large (C=96): use BLOCK_C=96 exactly to avoid masking waste
  - For tiny (C=16): BLOCK_C=16 exact, no masking
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
#  Tiny model: C=16, N=256
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({}, num_warps=1),
        triton.Config({}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def _ln_tiny_kernel(
    x_ptr,    # [1, C, H, W] NCHW  — elem [0,c,n] at c*N + n
    w_ptr,    # LN weight [C]
    b_ptr,    # LN bias [C]
    out_ptr,  # [1, N, C] contiguous
    N,
    C: tl.constexpr,   # 16
    eps,
):
    n     = tl.program_id(0)
    c_off = tl.arange(0, C)          # C=16 is power-of-2, no masking needed

    # Load weights first — they are small (16×2=32 bytes) and likely L1-cached
    ln_w = tl.load(w_ptr + c_off).to(tl.float32)
    ln_b = tl.load(b_ptr + c_off).to(tl.float32)

    x_raw = tl.load(x_ptr + c_off * N + n)
    x     = x_raw.to(tl.float32)

    mean    = tl.sum(x, 0) * (1.0 / C)
    diff    = x - mean
    var     = tl.sum(diff * diff, 0) * (1.0 / C)
    inv_std = 1.0 / tl.sqrt(var + eps)

    out  = (diff * inv_std * ln_w + ln_b).to(x_raw.dtype)

    tl.store(out_ptr + n * C + c_off, out)


def fused_swin_tiny(conv2d_out, ln_weight, ln_bias):
    B, C, N = 1, 16, 16 * 16   # N = 256
    out = torch.empty((B, N, C), device=conv2d_out.device, dtype=conv2d_out.dtype)
    _ln_tiny_kernel[(N,)](conv2d_out, ln_weight, ln_bias, out, N, C, 1e-5)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Large model: C=96, N=65536
#  BLOCK_C=128 (next power-of-2 >= 96) with element masking
# ─────────────────────────────────────────────────────────────────────────────

_BLOCK_C_LARGE = 128   # must be power-of-2 for tl.arange


@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2),
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _ln_large_kernel(
    x_ptr,     # [1, C, H, W] NCHW  — elem [0,c,n] at c*N + n
    w_ptr,     # LN weight [C]
    b_ptr,     # LN bias [C]
    out_ptr,   # [1, N, C] contiguous
    N,
    C: tl.constexpr,      # 96
    BLOCK_C: tl.constexpr, # 128
    eps,
):
    n      = tl.program_id(0)
    c_off  = tl.arange(0, BLOCK_C)    # 0..127
    c_mask = c_off < C                 # valid for 0..95

    # Load weights early so they hit L1 cache while NCHW data is in-flight
    ln_w = tl.load(w_ptr + c_off, mask=c_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(b_ptr + c_off, mask=c_mask, other=0.0).to(tl.float32)

    # Load main data (strided NCHW access)
    x_raw = tl.load(x_ptr + c_off * N + n, mask=c_mask, other=0.0)
    x     = x_raw.to(tl.float32)

    # masked zeros don't affect mean (sum / C with correct divisor)
    mean = tl.sum(x, 0) * (1.0 / C)
    # zero out masked slots before computing variance
    diff = tl.where(c_mask, x - mean, 0.0)
    var  = tl.sum(diff * diff, 0) * (1.0 / C)
    inv_std = 1.0 / tl.sqrt(var + eps)

    out  = (diff * inv_std * ln_w + ln_b).to(x_raw.dtype)

    tl.store(out_ptr + n * C + c_off, out, mask=c_mask)


def fused_swin_large(conv2d_out, ln_weight, ln_bias):
    B, C, N = 1, 96, 256 * 256   # N = 65536
    out = torch.empty((B, N, C), device=conv2d_out.device, dtype=conv2d_out.dtype)
    _ln_large_kernel[(N,)](conv2d_out, ln_weight, ln_bias, out, N, C, _BLOCK_C_LARGE, 1e-5)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Shared dispatch wrapper  (both pass files return this SAME object)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_swin_patch_embed(conv2d_out, ln_weight, ln_bias, route):
    """Route-dispatched replacement.  'route' is a string constant set by
    each pass's replacement_args()."""
    if route == "tiny":
        return fused_swin_tiny(conv2d_out, ln_weight, ln_bias)
    else:   # "large"
        return fused_swin_large(conv2d_out, ln_weight, ln_bias)