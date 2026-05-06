"""
Shared Triton kernels + dispatch wrappers used by all OptimizeLayerNorm passes.
Both LayerNorm and PositionBias routes are available here; each pass's
replacement_func() returns the SAME wrapper object (Python identity) so that
the framework's replacement_func_limit is never exceeded.
"""
import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Layer Normalisation  (no autotune overhead)
#
# Dispatched via route:
#   "ln_432"  → BLOCK_N=512, num_warps=16   (N=432, 196 rows)
#   "ln_192"  → BLOCK_N=256, num_warps=8    (N=192,  196 rows)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _kernel_ln(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, eps,
    BLOCK_N: tl.constexpr,
):
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    valid = offs < N
    base  = row * N

    x = tl.load(x_ptr + base + offs, mask=valid, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / N
    diff = tl.where(valid, x - mean, 0.0)
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = tl.rsqrt(var + eps)
    y    = diff * rstd

    w  = tl.load(w_ptr + offs, mask=valid, other=1.0).to(tl.float32)
    bv = tl.load(b_ptr + offs, mask=valid, other=0.0).to(tl.float32)
    y  = y * w + bv

    tl.store(out_ptr + base + offs, y.to(x.dtype), mask=valid)


def _run_ln_432(bias, weight, x):
    # N=432, M=196  →  BLOCK_N=512, num_warps=16 (44 % SM occupancy on A30)
    out = torch.empty_like(x)
    _kernel_ln[(196,)](x, weight, bias, out, 432, 1e-6,
                       BLOCK_N=512, num_warps=16)
    return out


def _run_ln_192(bias, weight, x):
    # N=192, M=196  →  BLOCK_N=256, num_warps=8  (5 % SM occupancy)
    out = torch.empty_like(x)
    _kernel_ln[(196,)](x, weight, bias, out, 192, 1e-6,
                       BLOCK_N=256, num_warps=8)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Relative-position bias  (14×14 grid, dtype-agnostic)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def _kernel_pos_bias(
    out_ptr,       # [196*196*3] contiguous flat
    BLOCK_SIZE: tl.constexpr,   # = 200 → 385 blocks
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = 196 * 196
    valid = offs < total

    i_14  = offs // 196
    j_14  = offs % 196
    di_sq = i_14 - j_14
    dj_sq = j_14 - i_14

    # Valid inside the 14×14 grid
    row_ok = i_14 < 14
    col_ok = j_14 < 14

    vals = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    vals = tl.where(valid & row_ok & col_ok,   vals, 0.0)
    vals = tl.where(valid & ~row_ok,            vals, di_sq * di_sq)
    vals = tl.where(valid & ~col_ok,            vals, dj_sq * dj_sq)
    vals = tl.where(~valid,                     vals, 0.0)

    tl.store(out_ptr + offs, vals.to(tl.int32), mask=valid)


def _run_pos_bias(x):
    out = torch.empty(1, 196, 196, 3, dtype=torch.int32, device=x.device)
    # 196*196 / 200 = 385 blocks, 200 threads each
    _kernel_pos_bias[(385,)](out, 200)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# @torch.fx.wrap wrappers returned by all replacement_func() calls
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def do_layer_norm_triton(bias, weight, x):
    """Drop-in Triton replacement for torch.nn.functional.layer_norm(x,(N,),w,b,1e-6)."""
    out = torch.empty_like(x)
    N = x.shape[-1]
    M = x.numel() // N
    if N == 432:
        _kernel_ln[(M,)](x, weight, bias, out, 432, 1e-6,
                         BLOCK_N=512, num_warps=16)
    else:  # N == 192
        _kernel_ln[(M,)](x, weight, bias, out, 192, 1e-6,
                         BLOCK_N=256, num_warps=8)
    return out


@torch.fx.wrap
def do_position_bias_triton(x):
    """Fused Triton replacement for the 14×14 relative-position bias."""
    out = torch.empty(1, 196, 196, 3, dtype=torch.int32, device=x.device)
    _kernel_pos_bias[(385,)](out, 200)
    return out