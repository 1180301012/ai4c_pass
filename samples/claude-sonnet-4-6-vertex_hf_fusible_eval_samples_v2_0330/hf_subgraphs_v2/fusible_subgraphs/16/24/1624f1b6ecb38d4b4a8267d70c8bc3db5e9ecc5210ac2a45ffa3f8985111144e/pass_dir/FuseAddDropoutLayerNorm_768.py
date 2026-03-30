import torch
import triton
import triton.language as tl


# ── Fused Add + LayerNorm kernel ──────────────────────────────────────────────
#
# Strategy:
#   N_COLS = 768 hardcoded as constexpr → tl.static_range for compile-time
#   unrolling; for CHUNK=256 (768/256=3) all chunks are full, so the compiler
#   eliminates all masking overhead entirely.
#
# Two passes per row:
#   Pass 1 – z = x+y, store tmp_12, accumulate Σz and Σz² (single-pass var)
#   Pass 2 – reload z, apply affine LN, store tmp_13
#
# Variance via E[X²]−E[X]² (float32 accum → numerically stable for fp16/bf16).

@triton.autotune(
    configs=[
        # CHUNK=256: 3 full iterations, zero wasted lanes (768=3×256 → no masking)
        triton.Config({'CHUNK': 256}, num_warps=2,  num_stages=1),
        triton.Config({'CHUNK': 256}, num_warps=4,  num_stages=1),
        triton.Config({'CHUNK': 256}, num_warps=8,  num_stages=1),
        triton.Config({'CHUNK': 256}, num_warps=16, num_stages=1),
        triton.Config({'CHUNK': 256}, num_warps=4,  num_stages=2),
        triton.Config({'CHUNK': 256}, num_warps=8,  num_stages=2),
        triton.Config({'CHUNK': 256}, num_warps=16, num_stages=2),
        # CHUNK=512: 2 iterations (second chunk has 25% masked lanes)
        triton.Config({'CHUNK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'CHUNK': 512}, num_warps=8,  num_stages=1),
        triton.Config({'CHUNK': 512}, num_warps=16, num_stages=1),
        triton.Config({'CHUNK': 512}, num_warps=4,  num_stages=2),
        triton.Config({'CHUNK': 512}, num_warps=8,  num_stages=2),
        # CHUNK=1024: single-pass, z stays in registers (no global reload), 25% waste
        triton.Config({'CHUNK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'CHUNK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'CHUNK': 1024}, num_warps=16, num_stages=1),
        triton.Config({'CHUNK': 1024}, num_warps=32, num_stages=1),
    ],
    key=['n_rows'],
)
@triton.jit
def fused_add_layernorm_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    ln_out_ptr,
    weight_ptr,
    bias_ptr,
    n_rows,
    eps,
    CHUNK: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    # Normalized dimension – fixed for this pattern (layer_norm shape = (768,))
    N_COLS: tl.constexpr = 768
    # Number of CHUNK-wide slices needed to cover N_COLS elements (compile-time)
    N_ITERS: tl.constexpr = (N_COLS + CHUNK - 1) // CHUNK

    row  = tl.program_id(0)
    base = row * N_COLS

    if N_ITERS == 1:
        # ── Single-pass path (CHUNK >= N_COLS, e.g. CHUNK=1024) ──────────────
        # z stays in registers the entire time → no global reload needed.
        cols = tl.arange(0, CHUNK)
        mask = cols < N_COLS

        x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
        y = tl.load(y_ptr + base + cols, mask=mask, other=0.0)
        z = x + y
        tl.store(out_ptr + base + cols, z, mask=mask)   # store tmp_12

        z_f    = tl.where(mask, z.to(tl.float32), 0.0)
        acc_s  = tl.sum(z_f,       axis=0)
        acc_sq = tl.sum(z_f * z_f, axis=0)
        mean   = acc_s  / N_COLS
        var    = acc_sq / N_COLS - mean * mean
        rstd   = tl.rsqrt(var + eps)

        w  = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b  = tl.load(bias_ptr   + cols, mask=mask, other=0.0).to(tl.float32)
        # z_f is still in registers – no reload!
        ln_f = (z_f - mean) * rstd * w + b

        if IS_BF16:
            tl.store(ln_out_ptr + base + cols, ln_f.to(tl.bfloat16), mask=mask)
        else:
            tl.store(ln_out_ptr + base + cols, ln_f.to(tl.float16),  mask=mask)

    else:
        # ── Multi-pass path (CHUNK < N_COLS, e.g. CHUNK=256 or 512) ─────────
        # Pass 1: z = x+y, store tmp_12, accumulate Σz and Σz² (single-pass var)
        # Pass 2: reload z from out_ptr, apply affine LN, store tmp_13
        acc_s  = tl.zeros([1], dtype=tl.float32)
        acc_sq = tl.zeros([1], dtype=tl.float32)

        for i in tl.static_range(N_ITERS):      # compile-time unrolled
            cols = i * CHUNK + tl.arange(0, CHUNK)
            mask = cols < N_COLS                 # compile-time True when CHUNK|N_COLS

            x = tl.load(x_ptr + base + cols, mask=mask, other=0.0)
            y = tl.load(y_ptr + base + cols, mask=mask, other=0.0)
            z = x + y
            tl.store(out_ptr + base + cols, z, mask=mask)

            z_f    = tl.where(mask, z.to(tl.float32), 0.0)
            acc_s  += tl.sum(z_f,       axis=0)
            acc_sq += tl.sum(z_f * z_f, axis=0)

        mean = acc_s  / N_COLS
        var  = acc_sq / N_COLS - mean * mean
        rstd = tl.rsqrt(var + eps)

        for i in tl.static_range(N_ITERS):
            cols = i * CHUNK + tl.arange(0, CHUNK)
            mask = cols < N_COLS

            z  = tl.load(out_ptr    + base + cols, mask=mask, other=0.0)
            w  = tl.load(weight_ptr + cols,        mask=mask, other=1.0).to(tl.float32)
            b  = tl.load(bias_ptr   + cols,        mask=mask, other=0.0).to(tl.float32)

            ln_f = (z.to(tl.float32) - mean) * rstd * w + b

            if IS_BF16:
                tl.store(ln_out_ptr + base + cols, ln_f.to(tl.bfloat16), mask=mask)
            else:
                tl.store(ln_out_ptr + base + cols, ln_f.to(tl.float16),  mask=mask)


@torch.fx.wrap
def _fused_add_layernorm_impl(a, b, weight, bias):
    """
    Actual Triton kernel dispatch – decorated @torch.fx.wrap so FX treats
    this as an opaque leaf node (returning a tuple) in the replacement graph.
    """
    n_cols = a.shape[-1]                # 768
    n_rows = a.numel() // n_cols        # 1 × 981 = 981

    out    = torch.empty_like(a)
    ln_out = torch.empty_like(a)

    IS_BF16 = (a.dtype == torch.bfloat16)

    fused_add_layernorm_kernel[(n_rows,)](
        a, b, out, ln_out,
        weight, bias,
        n_rows,
        1e-6,
        IS_BF16=IS_BF16,
    )

    return out, ln_out


def _fused_add_layernorm_replacement(a, b, weight, bias):
    """
    NOT @torch.fx.wrap — FX traces this function and creates:
      p0 = call_function: _fused_add_layernorm_impl(a, b, weight, bias)
      p1 = call_function: operator.getitem(p0, 0)   ← tmp_12
      p2 = call_function: operator.getitem(p0, 1)   ← tmp_13
      output: (p1, p2)
    Two explicit returning nodes → matches the pattern's two returning nodes.
    """
    result = _fused_add_layernorm_impl(a, b, weight, bias)
    return result[0], result[1]


# ─── Pattern / replacement glue ──────────────────────────────────────────────

def pattern(a, b, weight, bias_ln):
    """
    Matches:
        tmp_11 = a + b
        tmp_12 = dropout(tmp_11, 0.0, False, False)   # identity (p=0, eval)
        tmp_13 = layer_norm(tmp_12, (768,), weight, bias_ln, 1e-06)
        return (tmp_12, tmp_13)
    """
    tmp_11 = a + b
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (768,), weight, bias_ln, 1e-06)
    return (tmp_12, tmp_13)


def replacement_args(a, b, weight, bias_ln):
    return (a, b, weight, bias_ln)


def replacement_func():
    return _fused_add_layernorm_replacement