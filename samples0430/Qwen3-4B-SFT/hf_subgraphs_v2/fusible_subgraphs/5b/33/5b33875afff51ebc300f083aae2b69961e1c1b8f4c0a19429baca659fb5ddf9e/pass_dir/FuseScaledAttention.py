import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matches the shared attention head computation in every target graph
#   in_0 / 8.0  +  in_2  →  softmax  →  dropout(training=False)  →
#   matmul(in_3)  →  permute(0,2,1,3)  →  contiguous()
#
# Note: the reshape of in_1 is an independent pass-through that stays in the
# graph untouched.  The replacement returns only tmp_6; tmp_7 is produced by
# the original reshape node and wired up automatically by the pattern matcher.
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim = -1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel
#
# Computes for each (batch b, head h, query q):
#   scores  = in_0[b,h,q,:] / scale + in_2[b,0,0,:]
#   attn    = softmax(scores, dim=-1)            (dropout is no-op at inference)
#   out[b,q,h,:] = attn @ in_3[b,h,:,:]         written back in permuted layout
#
# Grid : (B * H * Sq)  — one program per output row
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_D': 8},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_D': 8},  num_warps=4, num_stages=2),
    ],
    key=['H', 'Sq', 'Sk', 'D'],
)
@triton.jit
def _fused_attn_kernel(
    # pointers
    in0_ptr,   # [B, H, Sq, Sk]  – attention scores
    in2_ptr,   # [B, 1, 1, Sk]   – attention mask
    in3_ptr,   # [B, H, Sk, D]   – value projections
    out_ptr,   # [B, Sq, H, D]   – output (permuted layout)
    # sizes
    B, H, Sq, Sk, D,
    scale,
    # strides
    in0_sb, in0_sh, in0_sq, in0_sk,
    in2_sb, in2_sk,
    in3_sb, in3_sh, in3_sk, in3_sd,
    # explicit strides used to make int64 arithmetic safe across types
    SO, SK,
    # tile sizes (constexpr)
    BLOCK_K: tl.constexpr,   # >= Sk, power-of-2
    BLOCK_D: tl.constexpr,   # >= D,  power-of-2
):
    # ── decode pid ──────────────────────────────────────────────────────────
    pid   = tl.program_id(0)
    m     = pid // (Sq * H)   # batch index
    rem   = pid % (Sq * H)
    q     = rem // H           # query index
    h_idx = rem % H            # head index

    k_offs = tl.arange(0, BLOCK_K)   # [0, 1, …, BLOCK_K-1]
    d_offs = tl.arange(0, BLOCK_D)   # [0, 1, …, BLOCK_D-1]
    k_mask = k_offs < Sk

    # ── load in_2[b, 0, 0, :] ──────────────────────────────────────────────
    # in_2 strides: (Sk, Sk, Sk, 1) for contiguous [B,1,1,Sk] layout
    in2_base = m * SO + k_offs * in2_sk   # pointer to [b, 0, 0, :]
    in2_vals = tl.load(in2_ptr + in2_base, mask=k_mask, other=0.0)

    # ── load in_0[b, h_idx, q, :] ───────────────────────────────────────────
    in0_base = (m * in0_sb + h_idx * in0_sh + q * in0_sq + k_offs * in0_sk)
    scores_f32 = tl.load(
        in0_ptr + in0_base,
        mask=k_mask, other=-1.0e9
    ).to(tl.float32)

    # ── fused scale + add mask ───────────────────────────────────────────────
    scores_f32 = scores_f32 * scale + in2_vals.to(tl.float32)

    # ── softmax over axis-0 (the Sk dimension) ───────────────────────────────
    m_max  = tl.max(scores_f32, axis=0)          # scalar
    exp_s  = tl.exp(scores_f32 - m_max)
    exp_s  = tl.where(k_mask, exp_s, 0.0)
    sum_exp = tl.sum(exp_s, axis=0)              # scalar
    attn   = exp_s / sum_exp                     # [BLOCK_K]

    # ── load in_3[b, h_idx, :, :] → [BLOCK_K, BLOCK_D] ─────────────────────
    in3_base  = m * in3_sb + h_idx * in3_sh
    w_base    = in3_base + k_offs[:, None] * in3_sk   # [BLOCK_K, 1]
    w_ptrs    = w_base + d_offs[None, :] * in3_sd     # [BLOCK_K, BLOCK_D]
    mask_2d   = k_mask[:, None]                        # [BLOCK_K, 1] → broadcasts
    w_vals    = tl.load(
        in3_ptr + w_ptrs,
        mask=mask_2d,
        other=0.0
    ).to(tl.float32)

    # ── matmul : [BLOCK_K] @ [BLOCK_K, BLOCK_D] = [BLOCK_D] ─────────────────
    out_f32 = tl.sum(attn[:, None] * w_vals, axis=0)  # [BLOCK_D]

    # ── store to out[b, q, h_idx, :] in the permuted [B, Sq, H, D] layout ───
    # out strides: (Sq*H*D, H*D, D, 1)
    out_base    = m * (Sq * H * D) + q * D + h_idx * D
    out_ptrs    = out_ptr + out_base + d_offs
    tl.store(out_ptrs, out_f32.to(out_ptr.dtype.element_ty))


# ──────────────────────────────────────────────────────────────────────────────
# Python wrapper (must be decorated with @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def fused_scaled_attention(in_0, in_2, in_3):
    """
    Fused implementation of:
        scaled = in_0 / 8.0 + in_2
        attn   = softmax(scaled, dim=-1)           (dropout is identity at inference)
        out    = (attn @ in_3).permute(0,2,1,3).contiguous()
        reshaped = reshape(in_1, [1,-1,H,D])
    """
    B, H, Sq, Sk = in_0.shape
    D = in_3.shape[3]

    out = torch.empty((B, Sq, H, D), dtype=in_0.dtype, device=in_0.device)

    # round up Sk and D to the next power-of-2 for the Triton tile sizes
    import math
    BLOCK_K = 1
    while BLOCK_K < Sk:
        BLOCK_K *= 2
    BLOCK_D = 1
    while BLOCK_D < D:
        BLOCK_D *= 2

    grid = (B * H * Sq,)

    _fused_attn_kernel[grid](
        in_0, in_2, in_3, out,
        B, H, Sq, Sk, D,
        8.0,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        Sq * Sk,
        Sk,
        BLOCK_K=BLOCK_K,
        BLOCK_D=BLOCK_D,
    )

    return out


def replacement_func():
    return fused_scaled_attention