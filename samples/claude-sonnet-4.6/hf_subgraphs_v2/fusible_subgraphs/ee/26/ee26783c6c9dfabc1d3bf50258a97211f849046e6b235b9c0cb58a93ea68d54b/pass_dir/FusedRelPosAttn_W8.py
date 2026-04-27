import torch
import triton
import triton.language as tl


# ── Triton kernel (W=8, H=8, SEQLEN=64, D=128, REL_LEN=15) ──────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 16}),
        triton.Config({'BLOCK_K': 32}),
        triton.Config({'BLOCK_K': 64}),
    ],
    key=['SEQLEN', 'D'],
)
@triton.jit
def _fused_rel_pos_attn_w8(
    in0_ptr,   # [B, SEQLEN, SEQLEN]
    in1_ptr,   # [B, H, W, D]
    in2_ptr,   # [B, W, H, W, W]
    in3_ptr,   # [D, REL_LEN]
    in4_ptr,   # [B, SEQLEN, D]
    out_ptr,   # [B, D, SEQLEN]
    B: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    D: tl.constexpr,
    SEQLEN: tl.constexpr,
    REL_LEN: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # One program per (batch, query)
    pid    = tl.program_id(0)
    bid    = pid // SEQLEN
    q_idx  = pid % SEQLEN

    r = q_idx // H   # row position  [0, W-1]
    h = q_idx % H    # head index    [0, H-1]

    D_range    = tl.arange(0, D)      # [D]
    k_range    = tl.arange(0, SEQLEN) # [SEQLEN]

    # ── Load query vector  in_1[bid, h, r, :]  shape [D] ──────────────────
    in1_base = bid * (H * W * D) + h * (W * D) + r * D
    q_vec = tl.load(in1_ptr + in1_base + D_range).to(tl.float32)

    # ── Load attention bias  in_0[bid, q_idx, :]  shape [SEQLEN] ───────────
    in0_base  = bid * (SEQLEN * SEQLEN) + q_idx * SEQLEN
    attn_bias = tl.load(in0_ptr + in0_base + k_range).to(tl.float32)

    # ── Load in_2[bid, r, h, :, :]  shape [W*W] (contiguous) ───────────────
    # in_2 is [B, W, H, W, W] contiguous
    in2_base = bid * (W * H * W * W) + r * (H * W * W) + h * (W * W)
    in2_vals = tl.load(in2_ptr + in2_base + k_range).to(tl.float32)

    # ── Compute relative position logits ────────────────────────────────────
    attn_scores = attn_bias + in2_vals

    for j in range(W):
        col_idx = (W - 1) + j - r       # scalar ∈ [0, 2W-2]
        col_vec = tl.load(in3_ptr + D_range * REL_LEN + col_idx).to(tl.float32)
        rel_logit_j = tl.sum(q_vec * col_vec)

        k_mask = (k_range >= j * W) & (k_range < (j + 1) * W)
        attn_scores = attn_scores + tl.where(k_mask, rel_logit_j, 0.0)

    # ── Softmax ────────────────────────────────────────────────────────────
    attn_max  = tl.max(attn_scores, axis=0)
    attn_exp  = tl.exp(attn_scores - attn_max)
    attn_sum  = tl.sum(attn_exp, axis=0)
    attn_prob = attn_exp / attn_sum   # [SEQLEN] fp32

    # ── Output matmul: out[bid, :, q_idx] = sum_k attn_prob[k] * in_4[bid,k,:] ─
    out_acc = tl.zeros([D], dtype=tl.float32)
    for k_start in range(0, SEQLEN, BLOCK_K):
        k_tile    = k_start + tl.arange(0, BLOCK_K)
        in4_ptrs  = in4_ptr + bid * (SEQLEN * D) + k_tile[:, None] * D + D_range[None, :]
        in4_block = tl.load(in4_ptrs).to(tl.float32)
        prob_tile = attn_prob[k_tile]
        out_acc   = out_acc + tl.sum(prob_tile[:, None] * in4_block, axis=0)

    # ── Store: out[bid, d, q_idx] ───────────────────────────────────────────
    out_dtype = out_acc.to(in0_ptr.dtype.element_ty)
    out_ptrs  = out_ptr + bid * (D * SEQLEN) + D_range * SEQLEN + q_idx
    tl.store(out_ptrs, out_dtype)


@torch.fx.wrap
def fused_rel_pos_attn_w8(in_0, in_1, in_2, in_3, in_4):
    B      = in_0.shape[0]   # 4
    SEQLEN = in_0.shape[1]   # 64
    H      = in_1.shape[1]   # 8
    W      = in_1.shape[2]   # 8
    D      = in_1.shape[3]   # 128
    REL_LEN = in_3.shape[1]  # 15

    out = torch.empty((B, D, SEQLEN), dtype=in_0.dtype, device=in_0.device)

    grid = (B * SEQLEN,)
    _fused_rel_pos_attn_w8[grid](
        in_0, in_1, in_2, in_3, in_4, out,
        B=B, H=H, W=W, D=D, SEQLEN=SEQLEN, REL_LEN=REL_LEN,
    )
    return (out,)


# ── Pattern / replacement API ────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4):
    matmul = in_1 @ in_3
    tmp_1  = matmul.reshape(-1, 8, 15)
    tmp_2  = torch.nn.functional.pad(tmp_1, [0, 1], 'constant', None)
    tmp_3  = tmp_2.flatten(1)
    tmp_4  = torch.nn.functional.pad(tmp_3, [0, 7], 'constant', None)
    tmp_5  = tmp_4.reshape(-1, 9, 15)
    tmp_6  = tmp_5[(slice(None, None, None), slice(None, 8, None), slice(7, None, None))]
    tmp_7  = tmp_6.reshape(4, 8, 1, 8, 8)
    tmp_8  = tmp_7.expand(-1, -1, 8, -1, -1)
    tmp_9  = tmp_8.permute((0, 3, 1, 4, 2))
    tmp_10 = tmp_9 + in_2
    tmp_11 = tmp_10.reshape(4, 64, 64)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return (tmp_15,)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_rel_pos_attn_w8