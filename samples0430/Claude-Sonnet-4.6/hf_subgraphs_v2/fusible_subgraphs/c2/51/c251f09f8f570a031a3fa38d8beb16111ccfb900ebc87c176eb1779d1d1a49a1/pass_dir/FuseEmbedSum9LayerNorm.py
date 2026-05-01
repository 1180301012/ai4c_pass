import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matches the sequential chain of 9 embedding-output additions
#           followed by layer_norm + dropout(training=False) that appears in
#           all LayoutLM embedding subgraphs.
# ---------------------------------------------------------------------------
def pattern(e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias):
    s1 = e1 + e2
    s2 = s1 + e3
    s3 = s2 + e4
    s4 = s3 + e5
    s5 = s4 + e6
    s6 = s5 + e7
    s7 = s6 + e8
    s8 = s7 + e9
    ln_out = torch.nn.functional.layer_norm(s8, (768,), ln_weight, ln_bias, 1e-12)
    dropout_out = torch.nn.functional.dropout(ln_out, 0.1, False, False)
    return dropout_out


def replacement_args(e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias):
    return (e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias)


# ---------------------------------------------------------------------------
# Triton kernel: fused sum-of-9-embeddings + LayerNorm
#
# Grid: (N_ROWS,) — one program per output row.
#
# e1/e9 : "large" tensors [batch, seq, 768] — streamed row by row.
# e2..e8: "small" broadcast tensors [1, seq, 768] — reused BATCH_SIZE times;
#          accessed via  (row % N_ROWS_SMALL),  fitting in L2 cache.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_sum9_layernorm_kernel(
    e1_ptr, e2_ptr, e3_ptr, e4_ptr, e5_ptr, e6_ptr, e7_ptr, e8_ptr, e9_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    N_ROWS,
    N_ROWS_SMALL,      # seq_len — for computing broadcast row index
    N_COLS: tl.constexpr,
    BLOCK:  tl.constexpr,
):
    row = tl.program_id(0)

    col      = tl.arange(0, BLOCK)
    mask     = col < N_COLS
    safe_col = tl.where(mask, col, 0)   # prevent OOB on masked lanes

    lb = row * N_COLS                          # large tensor base offset
    sb = (row % N_ROWS_SMALL) * N_COLS         # broadcast tensor base offset

    # ------------------------------------------------------------------
    # 1.  Load 9 embedding rows and accumulate in NATIVE dtype (fp16/bf16/fp32).
    #     Avoids 9 individual fp16→fp32 conversions; one conversion before LN.
    #     Masked lanes get other=0.0 (fp16/bf16/fp32 zero) so they contribute
    #     nothing to the mean/variance.
    # ------------------------------------------------------------------
    x  = tl.load(e1_ptr + lb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e2_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e3_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e4_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e5_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e6_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e7_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e8_ptr + sb + safe_col, mask=mask, other=0.0)
    x  = x + tl.load(e9_ptr + lb + safe_col, mask=mask, other=0.0)

    # Convert once to fp32 for numerically-stable LayerNorm
    xf = x.to(tl.float32)

    # ------------------------------------------------------------------
    # 2.  LayerNorm in fp32 using E[X²] − E[X]² variance formula.
    #     Avoids storing the 'diff' array: saves registers, improves
    #     instruction scheduling and occupancy.
    # ------------------------------------------------------------------
    # xf is already 0 for masked columns (other=0.0), so no tl.where needed
    # for the sums.
    s1 = tl.sum(xf, axis=0) / N_COLS           # mean   = E[X]
    s2 = tl.sum(xf * xf, axis=0) / N_COLS      # E[X²]
    # Clamp to 0 to guard against tiny negative values from fp cancellation.
    var  = tl.maximum(s2 - s1 * s1, 0.0)
    rstd = 1.0 / tl.sqrt(var + 1e-12)

    w  = tl.load(weight_ptr + safe_col, mask=mask, other=1.0).to(tl.float32)
    b  = tl.load(bias_ptr   + safe_col, mask=mask, other=0.0).to(tl.float32)
    # Apply affine; masked lanes normalised to 0 before mul → b at those spots,
    # but the store mask discards them anyway.
    out = tl.where(mask, xf - s1, 0.0) * rstd * w + b

    # ------------------------------------------------------------------
    # 3.  Store (Triton auto-converts fp32 → output-pointer dtype)
    # ------------------------------------------------------------------
    tl.store(out_ptr + lb + safe_col, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (opaque to FX tracing via @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_sum9_layernorm(e1, e2, e3, e4, e5, e6, e7, e8, e9, ln_weight, ln_bias):
    N_COLS       = 768
    N_ROWS       = e1.numel() // N_COLS
    N_ROWS_SMALL = e2.numel() // N_COLS

    out = torch.empty_like(e1)

    _fused_sum9_layernorm_kernel[(N_ROWS,)](
        e1, e2, e3, e4, e5, e6, e7, e8, e9,
        ln_weight, ln_bias,
        out,
        N_ROWS=N_ROWS,
        N_ROWS_SMALL=N_ROWS_SMALL,
        N_COLS=N_COLS,
        BLOCK=1024,
        num_warps=8,
        num_stages=2,
    )

    return out


# ---------------------------------------------------------------------------
# Required: zero-argument factory that returns the callable
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_sum9_layernorm