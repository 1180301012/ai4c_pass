"""
Fused pass: add + layer_norm + reshape/permute/contiguous/permute/reshape

Optimisation strategy for tiny tensor [1, 4, 128]:
  • Fuse element-wise add, layer-norm, and the contiguous() copy into ONE
    Triton kernel call.
  • Single block (grid=(1,)), compile-time-unrolled loop over N=4 rows.
    w/b loaded once before the loop and kept in registers for all rows.
  • num_warps=2 (64 threads, 2 elements per thread): 2 warps can hide each
    other's memory latency while requiring only 1 cross-warp sync per
    reduction (5 intra + 1 cross = 6 shuffles vs 5 for num_warps=1).
  • Two-pass per row: mean then var (avoids numerical issues of E[x²]-E[x]²).
  • No @triton.autotune.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel  – single block, unrolled loop, 2 warps
# ---------------------------------------------------------------------------

@triton.jit
def fused_add_ln_kernel(
    x_ptr,          # [total_rows, C] contiguous
    res_ptr,        # [total_rows, C] contiguous
    w_ptr,          # [C] gamma
    b_ptr,          # [C] beta
    out_ptr,        # [total_rows, C] contiguous output
    C,              # channels = 128
    eps,
    N_ROWS: tl.constexpr,   # 4
    BLOCK_C: tl.constexpr,  # 128
    IS_FP16: tl.constexpr,
):
    c_offs = tl.arange(0, BLOCK_C)
    mask   = c_offs < C

    # Load gamma/beta once – stay in registers across the whole loop
    w = tl.load(w_ptr + c_offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + c_offs, mask=mask, other=0.0).to(tl.float32)

    for n in tl.static_range(N_ROWS):
        base = n * C
        x   = tl.load(x_ptr   + base + c_offs, mask=mask, other=0.0).to(tl.float32)
        res = tl.load(res_ptr  + base + c_offs, mask=mask, other=0.0).to(tl.float32)

        y     = x + res
        mean  = tl.sum(y, axis=0) / C
        diff  = y - mean
        var   = tl.sum(diff * diff, axis=0) / C
        rstd  = tl.rsqrt(var + eps)
        y_out = diff * rstd * w + b

        if IS_FP16:
            tl.store(out_ptr + base + c_offs, y_out.to(tl.float16),  mask=mask)
        else:
            tl.store(out_ptr + base + c_offs, y_out.to(tl.bfloat16), mask=mask)


# ---------------------------------------------------------------------------
# FX-visible wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_add_layernorm_permute(bias, weight, x, residual):
    B, N, C  = x.shape          # 1, 4, 128
    total_rows = B * N           # 4

    out_flat = torch.empty(total_rows * C, dtype=x.dtype, device=x.device)
    x_flat   = x.reshape(total_rows, C)
    res_flat = residual.reshape(total_rows, C)

    IS_FP16 = (x.dtype == torch.float16)

    # Single block, 1 warp, compile-time-unrolled loop over 4 rows.
    # w/b cached in registers; pure intra-warp reductions.
    fused_add_ln_kernel[(1,)](
        x_flat, res_flat, weight, bias, out_flat,
        C,
        1e-5,
        N_ROWS=total_rows,
        BLOCK_C=128,
        IS_FP16=IS_FP16,
        num_warps=1,
    )

    return out_flat.reshape(B, N, C)


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    """Mirrors model.py forward exactly (same op variants, same arg order)."""
    tmp_2 = in_3 + in_2
    tmp_3 = torch.nn.functional.layer_norm(tmp_2, (128,), in_1, in_0, 1e-05)
    tmp_4 = tmp_3.reshape(1, 2, 2, -1)
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = tmp_6.permute(0, 2, 3, 1)
    tmp_8 = tmp_7.reshape(1, -1, 128)
    return tmp_8


def replacement_args(in_0, in_1, in_2, in_3):
    """Return args in the same positional order as pattern parameters."""
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    """Return the wrapper function object (zero-arg factory)."""
    return fused_add_layernorm_permute