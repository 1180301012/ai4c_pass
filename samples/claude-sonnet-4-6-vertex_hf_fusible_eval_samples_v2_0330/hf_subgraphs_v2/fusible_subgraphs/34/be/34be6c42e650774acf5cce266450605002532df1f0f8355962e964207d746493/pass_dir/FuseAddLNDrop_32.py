import torch
import triton
import triton.language as tl


# ─── Pattern ─────────────────────────────────────────────────────────────────
# Matches: add(word_emb, pos_embed)  →  layer_norm((32,))  →  dropout(noop)

def pattern(word_emb, pos_embed, in_2, in_1):
    tmp_16 = word_emb + pos_embed
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(word_emb, pos_embed, in_2, in_1):
    return (word_emb, pos_embed, in_2, in_1)


# ─── Triton kernel ───────────────────────────────────────────────────────────
# No autotune: fixed BLOCK_H/num_warps to minimise per-call Python overhead.
# H=32 fits exactly in BLOCK_H=32: no masking needed, one warp per row.

@triton.jit
def _add_ln_32_kernel(
    wemb_ptr,       # [..., 32] word embeddings  (bf16/fp16, contiguous)
    pemb_ptr,       # [..., 32] position embeds  (bf16/fp16, contiguous)
    weight_ptr,     # [32]      LN weight         (bf16/fp16)
    bias_ptr,       # [32]      LN bias           (bf16/fp16)
    out_ptr,        # [..., 32] output            (bf16/fp16, contiguous)
    N,
    eps,
    BLOCK_H: tl.constexpr,   # 32 for H=32: no padding/masking
):
    H    = 32
    row  = tl.program_id(0)
    cols = tl.arange(0, BLOCK_H)   # 0..31; BLOCK_H == H → no mask needed

    # ── Load and upcast to fp32 ──
    w = tl.load(wemb_ptr   + row * H + cols).to(tl.float32)
    p = tl.load(pemb_ptr   + row * H + cols).to(tl.float32)
    g = tl.load(weight_ptr + cols).to(tl.float32)
    b = tl.load(bias_ptr   + cols).to(tl.float32)

    # ── Fused add + single-pass layer-norm ──
    x    = w + p
    mean = tl.sum(x, axis=0) / H
    x_c  = x - mean
    var  = tl.sum(x_c * x_c, axis=0) / H
    rstd = tl.rsqrt(var + eps)
    out  = x_c * rstd * g + b

    # ── Store (Triton auto-casts fp32 → bf16/fp16 based on out_ptr dtype) ──
    tl.store(out_ptr + row * H + cols, out)


# ─── Python wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_add_ln_drop_32(word_emb, pos_embed, in_2, in_1):
    """
    word_emb  : [..., 32]  bf16/fp16
    pos_embed : [..., 32]  bf16/fp16
    in_2      : [32]       bf16/fp16  layer-norm weight
    in_1      : [32]       bf16/fp16  layer-norm bias
    returns   : same shape and dtype as word_emb
    """
    H   = 32
    N   = word_emb.numel() // H   # B * S  (= 15 in target graphs)
    eps = 1e-5
    out = torch.empty_like(word_emb)

    _add_ln_32_kernel[(N,)](
        word_emb, pos_embed, in_2, in_1, out,
        N, eps,
        BLOCK_H=32,
        num_warps=1,
    )

    return out


# ─── replacement_func ────────────────────────────────────────────────────────

def replacement_func():
    return fused_add_ln_drop_32