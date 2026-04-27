"""
Shared Triton kernels + single dispatch wrapper for LayerNorm variants.
Single-output pattern (only LN) avoids the dual-output SubgraphRewriter crash.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pure layer-norm kernel  (x is tmp_8, already computed by the add)
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _ln_kernel(
    x_ptr,        # input  (1, N, C)  — the residual sum tmp_8
    weight_ptr,   # LN weight (C,)
    bias_ptr,     # LN bias   (C,)
    out_ptr,      # output (1, N, C)  — tmp_9
    eps,
    C:       tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid    = tl.program_id(0)
    base   = pid * C
    c_offs = tl.arange(0, BLOCK_C)
    mask   = c_offs < C

    x = tl.load(x_ptr + base + c_offs, mask=mask, other=0.0)

    xf   = x.to(tl.float32)
    xfm  = tl.where(mask, xf, 0.0)
    mean = tl.sum(xfm, axis=0) / C

    diff  = xf - mean
    diffm = tl.where(mask, diff, 0.0)
    var   = tl.sum(diffm * diffm, axis=0) / C
    rstd  = tl.rsqrt(var + eps)
    norm  = diff * rstd

    w      = tl.load(weight_ptr + c_offs, mask=mask, other=0.0).to(tl.float32)
    bv     = tl.load(bias_ptr   + c_offs, mask=mask, other=0.0).to(tl.float32)
    ln_out = (norm * w + bv).to(x.dtype)

    tl.store(out_ptr + base + c_offs, ln_out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Single @torch.fx.wrap dispatcher  (same object in every pass file)
# N: 16384→C=96 | 4096→C=192 | 1024→C=384
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_ln(x, weight, bias, N):
    if N == 16384:
        C, BLOCK_C, NW = 96,  128, 4
    elif N == 4096:
        C, BLOCK_C, NW = 192, 256, 4
    else:              # N == 1024
        C, BLOCK_C, NW = 384, 512, 8

    out = torch.empty(1, N, C, dtype=x.dtype, device=x.device)

    _ln_kernel[(N,)](
        x, weight, bias, out,
        eps=1e-5,
        C=C, BLOCK_C=BLOCK_C,
        num_warps=NW,
    )

    return out