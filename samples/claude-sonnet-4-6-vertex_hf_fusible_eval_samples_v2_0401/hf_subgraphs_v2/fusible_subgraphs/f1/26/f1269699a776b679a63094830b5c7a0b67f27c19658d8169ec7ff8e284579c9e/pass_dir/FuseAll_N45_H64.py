import torch
import triton
import triton.language as tl


# ── Precompute constant relative-position-bias (N=45) at import time ──────────
def _make_rel_pos_bias_45():
    N = 45
    tmp_10 = torch.arange(N, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(N, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 = tmp_19 + tmp_31
    return tmp_19

_REL_POS_BIAS_45 = _make_rel_pos_bias_45()


# ── Pattern: entire forward for N=45, H=64, eps=1e-12 ────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_5 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_6 = torch.nn.functional.embedding(in_5, in_3, 1, None, 2.0, False, False)
    tmp_7 = tmp_5 + tmp_6
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (64,), in_2, in_1, 1e-12)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.1, False, False)
    tmp_10 = torch.arange(45, dtype=torch.int64)
    tmp_11 = tmp_10[(slice(None, None, None), None)]
    tmp_12 = torch.arange(45, dtype=torch.int64)
    tmp_13 = tmp_12[(None, slice(None, None, None))]
    tmp_14 = tmp_13 - tmp_11
    tmp_15 = -tmp_14
    tmp_16 = tmp_15 < 0
    tmp_17 = tmp_16.to(torch.int64)
    tmp_18 = tmp_17 * 16
    tmp_19 = 0 + tmp_18
    tmp_20 = torch.abs(tmp_15)
    tmp_21 = tmp_20 < 8
    tmp_22 = tmp_20.float()
    tmp_23 = tmp_22 / 8
    tmp_24 = torch.log(tmp_23)
    tmp_25 = tmp_24 / 2.772588722239781
    tmp_26 = tmp_25 * 8
    tmp_27 = tmp_26.to(torch.int64)
    tmp_28 = 8 + tmp_27
    tmp_29 = torch.full_like(tmp_28, 15)
    tmp_30 = torch.min(tmp_28, tmp_29)
    tmp_31 = torch.where(tmp_21, tmp_20, tmp_30)
    tmp_19 += tmp_31
    tmp_32 = tmp_19
    return (tmp_9, tmp_32)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# ── Triton kernel: embedding-add-layernorm (H=64, eps=1e-12) ──────────────────
@triton.jit
def emb_add_ln_h64_kernel(
    idx1_ptr, emb1_ptr,
    idx2_ptr, emb2_ptr,
    weight_ptr, bias_ptr,
    out_ptr,
    BLOCK_H: tl.constexpr,
):
    H = 64
    eps = 1e-12

    pid = tl.program_id(0)
    idx1 = tl.load(idx1_ptr + pid)
    idx2 = tl.load(idx2_ptr + pid)

    h_offs = tl.arange(0, BLOCK_H)
    # BLOCK_H == H == 64: mask is always True, no masking overhead

    e1 = tl.load(emb1_ptr + idx1 * H + h_offs).to(tl.float32)
    e2 = tl.load(emb2_ptr + idx2 * H + h_offs).to(tl.float32)
    x = e1 + e2

    mean = tl.sum(x, axis=0) / H
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / H
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    w = tl.load(weight_ptr + h_offs).to(tl.float32)
    b = tl.load(bias_ptr + h_offs).to(tl.float32)
    out = x_norm * w + b

    tl.store(out_ptr + pid * H + h_offs, out)


# ── Replacement wrapper ───────────────────────────────────────────────────────
@torch.fx.wrap
def fused_all_n45_h64(in_0, in_1, in_2, in_3, in_4, in_5):
    H = 64
    B, L = in_0.shape
    N_tok = B * L

    orig_dtype = in_4.dtype
    device = in_4.device

    idx1 = in_0.reshape(-1).contiguous()
    idx2 = in_5.reshape(-1).contiguous()
    emb1 = in_4.contiguous()
    emb2 = in_3.contiguous()
    weight = in_2.to(torch.float32).contiguous()
    bias = in_1.to(torch.float32).contiguous()

    out_f32 = torch.empty(N_tok, H, dtype=torch.float32, device=device)

    emb_add_ln_h64_kernel[(N_tok,)](
        idx1, emb1, idx2, emb2, weight, bias, out_f32,
        BLOCK_H=64,
    )

    result = out_f32.reshape(B, L, H)
    if orig_dtype != torch.float32:
        result = result.to(orig_dtype)

    # Return precomputed constant relative-position-bias (CPU int64 [45,45])
    return (result, _REL_POS_BIAS_45)


def replacement_func():
    return fused_all_n45_h64