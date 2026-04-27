import torch
import triton
import triton.language as tl


def pattern(in0, in1, in2, in3, in4, in5, in6, in7, in8, ln_weight, ln_bias):
    t1 = in0 + in1
    t2 = t1  + in2
    t3 = t2  + in3
    t4 = t3  + in4
    t5 = t4  + in5
    t6 = t5  + in6
    t7 = t6  + in7
    t8 = t7  + in8
    t9  = torch.nn.functional.layer_norm(t8, (768,), ln_weight, ln_bias, 1e-12)
    out = torch.nn.functional.dropout(t9, 0.1, False, False)
    return out


def replacement_args(in0, in1, in2, in3, in4, in5, in6, in7, in8, ln_weight, ln_bias):
    return (in0, in1, in2, in3, in4, in5, in6, in7, in8, ln_weight, ln_bias)


# ─────────────────────────────────────────────────────────────────────────────
# Broadcast layout:
#   in0/in8   shape [B, S, D]  → row i → offset  i * D
#   in1..in7  shape [1, S, D]  → row i → offset (i % N1) * D
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _layoutlm_fused_kernel(
    e0_ptr, e1_ptr, e2_ptr, e3_ptr, e4_ptr,
    e5_ptr, e6_ptr, e7_ptr, e8_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    N,
    N1,
    BLOCK_D: tl.constexpr,
    D:       tl.constexpr,
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_D)
    mask = col_offsets < D

    base_b = row_idx * D
    base_1 = (row_idx % N1) * D

    v0 = tl.load(e0_ptr + base_b + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(e1_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(e2_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(e3_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v4 = tl.load(e4_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v5 = tl.load(e5_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v6 = tl.load(e6_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v7 = tl.load(e7_ptr + base_1 + col_offsets, mask=mask, other=0.0).to(tl.float32)
    v8 = tl.load(e8_ptr + base_b + col_offsets, mask=mask, other=0.0).to(tl.float32)

    x = v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7 + v8

    mean  = tl.sum(x, axis=0) / D
    x_m   = x - mean
    var   = (tl.sum(x_m * x_m, axis=0) - (BLOCK_D - D) * mean * mean) / D
    rstd  = tl.math.rsqrt(var + 1e-12)

    w     = tl.load(weight_ptr + col_offsets, mask=mask, other=0.0).to(tl.float32)
    b_val = tl.load(bias_ptr   + col_offsets, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + base_b + col_offsets, x_m * rstd * w + b_val, mask=mask)


@torch.fx.wrap
def layoutlm_fused_emb_sum_layernorm(in0, in1, in2, in3, in4, in5, in6, in7, in8,
                                      ln_weight, ln_bias):
    D  = in0.shape[-1]
    N  = in0.numel() // D
    N1 = in1.numel() // D
    out = torch.empty_like(in0)
    _layoutlm_fused_kernel[(N,)](
        in0, in1, in2, in3, in4, in5, in6, in7, in8,
        ln_weight, ln_bias,
        out,
        N, N1,
        BLOCK_D=1024,
        D=768,
        num_warps=8,
    )
    return out


def replacement_func():
    return layoutlm_fused_emb_sum_layernorm