import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_3 = in_3 + in_2
    tmp_4 = tmp_3.float()
    tmp_5 = tmp_4.mean(-1, keepdim=True)
    tmp_6 = tmp_4 - tmp_5
    tmp_7 = tmp_6.pow(2)
    tmp_8 = tmp_7.mean(-1, keepdim=True)
    tmp_9 = tmp_4 - tmp_5
    tmp_10 = tmp_8 + 1e-07
    tmp_11 = torch.sqrt(tmp_10)
    tmp_12 = tmp_9 / tmp_11
    tmp_13 = tmp_12.to(torch.float32)
    tmp_14 = in_1 * tmp_13
    tmp_15 = tmp_14 + in_0
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def _fused_add_layernorm_kernel(
    in2_ptr,
    in3_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    N: tl.constexpr,     # always 768 — compile-time constant → mask is compile-time
    BLOCK_SIZE: tl.constexpr,
):
    """One Triton block per row: fused (in3+in2) + LayerNorm(weight, bias).

    One-pass variance formula  var = E[z²] − E[z]²
    ─ Masked positions are loaded as 0.0 so they contribute exactly 0 to
      both sum(z) and sum(z²) — no tl.where needed.
    ─ Fused affine: (z−mean)*rstd*w + b  ≡  z*(rstd·w) + (b − mean·rstd·w)
      avoids materialising a separate `diff` array.
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N          # compile-time constant when N is tl.constexpr

    # ── Load residual inputs (bf16/fp16/fp32 → fp32) ─────────────────────
    x = tl.load(in2_ptr + row_idx * N + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(in3_ptr + row_idx * N + offsets, mask=mask, other=0.0).to(tl.float32)
    z = x + y   # masked lanes are exactly 0 (from other=0.0)

    # ── One-pass statistics (masked 0s contribute 0 to both sums) ────────
    sum_z    = tl.sum(z,     axis=0)
    sum_z_sq = tl.sum(z * z, axis=0)
    mean = sum_z / N
    var  = sum_z_sq / N - mean * mean   # E[z²] − E[z]²
    rstd = tl.rsqrt(tl.maximum(var, 0.0) + 1e-07)

    # ── Load affine parameters ────────────────────────────────────────────
    w = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)

    # ── Fused normalise + scale + shift (no diff array) ──────────────────
    rw  = rstd * w                   # per-element scale factor
    out = z * rw + (b - mean * rw)   # (z − mean) * rstd * w + b

    tl.store(out_ptr + row_idx * N + offsets, out, mask=mask)


@torch.fx.wrap
def fused_add_layer_norm(in_0, in_1, in_2, in_3):
    """Wrapper: flatten to 2-D rows, launch one block per row, reshape back.

    Fixed config, no autotune overhead: BLOCK_SIZE=1024, num_warps=8, num_stages=1.
    NOTE: use torch.empty with explicit dtype=float32 (NOT empty_like) so that
    the FX dtype-inference sees a float32 output regardless of in_2's dtype.
    """
    shape = in_2.shape
    N = shape[-1]           # 768
    M = in_2.numel() // N

    out = torch.empty((M, N), dtype=torch.float32, device=in_2.device)

    _fused_add_layernorm_kernel[(M,)](
        in_2, in_3, in_1, in_0, out, N,
        BLOCK_SIZE=1024, num_warps=4, num_stages=1,
    )

    return out.view(shape)


def replacement_func():
    return fused_add_layer_norm