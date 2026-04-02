import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match  gelu → slice
#                            \
#                  slice      add → transpose → layer_norm → dropout
#                /
#  avg_pool_out
# ---------------------------------------------------------------------------

def pattern(conv_out, avg_pool_out, weight, bias):
    # conv_out    : [B, C, Tx]  (conv1d output, Tx=125)
    # avg_pool_out: [B, C, T ]  (avg_pool1d output, T=124)
    gelu_out   = torch.nn.functional.gelu(conv_out)
    sliced_avg  = avg_pool_out[(Ellipsis, slice(None, 124, None))]
    sliced_gelu = gelu_out[(Ellipsis, slice(None, 124, None))]
    added       = sliced_avg + sliced_gelu
    transposed  = added.transpose(1, 2)
    normalized  = torch.nn.functional.layer_norm(transposed, (768,), weight, bias, 1e-05)
    result      = torch.nn.functional.dropout(normalized, 0.1, False, False)
    return result


def replacement_args(conv_out, avg_pool_out, weight, bias):
    return (conv_out, avg_pool_out, weight, bias)


# ---------------------------------------------------------------------------
# Triton kernel — one program per (batch, time-step)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 1024}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_C': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_C': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 2048}, num_warps=16, num_stages=1),
    ],
    key=['C', 'T', 'Tx'],
)
@triton.jit
def fused_gelu_slice_add_ln_kernel(
    conv_ptr,   # [B, C, Tx]  conv1d output (pre-GELU)
    avg_ptr,    # [B, C, T ]  avg_pool output
    w_ptr,      # [C]          layer-norm weight
    b_ptr,      # [C]          layer-norm bias
    out_ptr,    # [B, T, C]   output (transposed + normalized)
    B,
    C,
    T,
    Tx,
    eps,
    BLOCK_C: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    bi  = pid // T
    t   = pid %  T

    offs_c = tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    # ---- Load conv[bi, :, t] and apply GELU(x) = x·0.5·(1+erf(x/√2)) ------
    conv_off = bi * C * Tx + offs_c * Tx + t
    x      = tl.load(conv_ptr + conv_off, mask=mask_c, other=0.0).to(tl.float32)
    gelu_x = x * 0.5 * (1.0 + tl.math.erf(x * 0.7071067811865476))

    # ---- Load avg_pool[bi, :, t] (slice :T is a no-op for T=124) -----------
    avg_off = bi * C * T + offs_c * T + t
    y = tl.load(avg_ptr + avg_off, mask=mask_c, other=0.0).to(tl.float32)

    # ---- Add ----------------------------------------------------------------
    s = gelu_x + y

    # ---- Layer-norm over C elements -----------------------------------------
    mean        = tl.sum(s, axis=0) / C
    diff        = s - mean
    diff_masked = tl.where(mask_c, diff, 0.0)          # zero BLOCK_C padding
    var         = tl.sum(diff_masked * diff_masked, axis=0) / C
    inv_std     = tl.rsqrt(var + eps)
    norm        = diff_masked * inv_std

    # ---- Affine transform ---------------------------------------------------
    wt    = tl.load(w_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    bias_ = tl.load(b_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    out   = norm * wt + bias_

    # ---- Store to out[bi, t, :] — contiguous in C --------------------------
    out_off = bi * T * C + t * C + offs_c
    if IS_BF16:
        tl.store(out_ptr + out_off, out.to(tl.bfloat16), mask=mask_c)
    else:
        tl.store(out_ptr + out_off, out.to(tl.float16),  mask=mask_c)


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_gelu_slice_add_transpose_ln_dropout(conv_out, avg_pool_out, weight, bias):
    B, C, Tx = conv_out.shape        # [1, 768, 125]
    T        = avg_pool_out.shape[2]  # 124

    out     = torch.empty(B, T, C, dtype=conv_out.dtype, device=conv_out.device)
    IS_BF16 = (conv_out.dtype == torch.bfloat16)

    grid = (B * T,)
    fused_gelu_slice_add_ln_kernel[grid](
        conv_out, avg_pool_out, weight, bias, out,
        B, C, T, Tx,
        1e-5,
        IS_BF16=IS_BF16,
    )
    return out


def replacement_func():
    return fused_gelu_slice_add_transpose_ln_dropout