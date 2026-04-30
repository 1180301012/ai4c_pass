import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Pattern: matches the full chain in model.py
#   multi_head_attention_forward → getitem[0] → dropout(p=0.0) → dropout(p=0.0)
# ──────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3, in_4):
    mha_out = torch.nn.functional.multi_head_attention_forward(
        in_4, in_4, in_4, 512, 8, in_3, in_2, None, None, False, 0.0, in_1, in_0,
        training=False, key_padding_mask=None, need_weights=True, attn_mask=None,
        average_attn_weights=True, is_causal=False
    )
    out = mha_out[0]
    out = torch.nn.functional.dropout(out, 0.0, False, False)
    out = torch.nn.functional.dropout(out, 0.0, False, False)
    return out


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused MHA with in/out projections
#   Grid: (cdiv(S, BLOCK_M) * B,  H)
#   Each block computes [BLOCK_M, D_h] of the output.
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=4, num_stages=3),
    ],
    key=['S', 'H'],
)
@triton.jit
def mha_fused_kernel(
    x_ptr,      # [B, S, D]  input
    wq_ptr,     # [3*D_h, D]  input projection weight
    bq_ptr,     # [3*D_h]     input projection bias
    wo_ptr,     # [D, D_h]    output projection weight
    bo_ptr,     # [D]         output projection bias
    out_ptr,    # [S, B, D]  output  (written as [B*S, D] in memory)
    # strides for x  (B, S, D)
    sx_b, sx_s, sx_d,
    # strides for wq (out_feat, in_feat)
    sw_o, sw_i,
    # strides for wo  (out_feat, in_feat)
    so_o, so_i,
    B, S, H,
    D_h: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Derived constants
    D: tl.constexpr = D_h * 8          # = 512
    Q_START: tl.constexpr = 0
    K_START: tl.constexpr = D_h * 8    # = 512
    V_START: tl.constexpr = D_h * 16   # = 1024

    pid_m  = tl.program_id(0)   # block along S
    head   = tl.program_id(1)   # which head
    batch  = tl.program_id(2)   # which batch

    # Row indices in the [B*S, D] output space
    q_row = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    d_idx = tl.arange(0, D_h)                          # [D_h]

    q_mask = q_row < S

    # ── Load Q [BLOCK_M, D_h] ──────────────────────────────────────────────
    x_base = batch * S * D + q_row[:, None] * D + d_idx[None, :]
    q = tl.load(x_ptr + x_base,
                mask=q_mask[:, None],
                other=0.0)
    if IS_FP16:
        q = q.to(tl.float16)
    elif IS_BF16:
        q = q.to(tl.bfloat16)

    # ── Attention accumulation ─────────────────────────────────────────────
    attn_out = tl.zeros((BLOCK_M, D_h), dtype=tl.float32)

    kv_base = batch * S * D + tl.arange(0, S)[:, None] * D + d_idx[None, :]

    for n0 in range(0, S, BLOCK_N):
        n_idx = n0 + tl.arange(0, BLOCK_N)           # [BLOCK_N]
        n_mask = n_idx < S

        # Load K [BLOCK_N, D_h]
        k = tl.load(x_ptr + kv_base + Q_START * D + n_idx[:, None] * D + d_idx[None, :],
                    mask=n_mask[:, None], other=0.0)
        if IS_FP16:
            k = k.to(tl.float16)
        elif IS_BF16:
            k = k.to(tl.bfloat16)

        # Attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k, out_dtype=tl.float32)
        scores = tl.where(n_mask[None, :], scores, float('-inf'))

        # Load V [BLOCK_N, D_h]
        v = tl.load(x_ptr + kv_base + V_START * D + n_idx[:, None] * D + d_idx[None, :],
                    mask=n_mask[:, None], other=0.0)
        if IS_FP16:
            v = v.to(tl.float16)
        elif IS_BF16:
            v = v.to(tl.bfloat16)

        # Softmax over key dimension
        max_score  = tl.max(scores, axis=1)           # [BLOCK_M]
        exp_scores = tl.exp(scores - max_score[:, None])
        exp_scores = tl.where(n_mask[None, :], exp_scores, 0.0)
        attn_weights = exp_scores / tl.sum(exp_scores, axis=1)[:, None]

        # Weighted sum [BLOCK_M, D_h]
        av = tl.dot(attn_weights.to(q.dtype), v, out_dtype=tl.float32)
        attn_out = attn_out + av

    # ── Output projection ──────────────────────────────────────────────────
    # out_ptr[batch * S * D + q_row, d] = wo[d_h, :] · attn_out + bo[d]
    # d ranges over [head*D_h : (head+1)*D_h]
    d_out = head * D_h + d_idx                         # [D_h]

    wo = tl.load(wo_ptr + d_out[:, None] * so_o + d_idx[None, :] * so_i)
    bo = tl.load(bo_ptr + d_out)

    result = tl.zeros((BLOCK_M, D_h), dtype=tl.float32)
    for i in tl.static_range(D_h):
        col = attn_out[:, i]
        result_i = tl.sum(col[:, None] * wo[i, :], axis=1) + bo[i]
        result = tl.where(d_idx[None, :] == i, result_i[:, None], result)

    if IS_FP16:
        result = result.to(tl.float16)
    elif IS_BF16:
        result = result.to(tl.bfloat16)

    # ── Store output ───────────────────────────────────────────────────────
    out_base = batch * S * D + q_row[:, None] * D + d_out[None, :]
    tl.store(out_ptr + out_base, result, mask=q_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Kernel wrapper  (must be @torch.fx.wrap)
# ──────────────────────────────────────────────────────────────────────────────
@torch.fx.wrap
def mha_triton_wrapper(in_0, in_1, in_2, in_3, in_4):
    """
    in_0 : out_proj_bias    [D]
    in_1 : out_proj_weight  [D, D_h]
    in_2 : in_proj_bias     [3*D_h]
    in_3 : in_proj_weight   [3*D_h, D]
    in_4 : x                [S, B, D]   (batch-1 dim squeezed above)
    """
    B = in_4.shape[1]   # 1
    S = in_4.shape[0]   # 150
    H = 8
    D_h = 64            # head_dim = 512 / 8

    out = torch.empty((S, B, 512), dtype=in_4.dtype, device=in_4.device)

    IS_FP16 = (in_4.dtype == torch.float16)
    IS_BF16 = (in_4.dtype == torch.bfloat16)

    grid = lambda meta: (triton.cdiv(S, meta['BLOCK_M']), H, B)

    mha_fused_kernel[grid](
        in_4,   # x
        in_3,   # wq  (in_proj_weight)
        in_2,   # bq  (in_proj_bias)
        in_1,   # wo  (out_proj_weight)
        in_0,   # bo  (out_proj_bias)
        out,
        # x strides
        in_4.stride(1), in_4.stride(2), in_4.stride(0),  # (S, D, ?)
        # wq strides
        in_3.stride(0), in_3.stride(1),
        # wo strides
        in_1.stride(0), in_1.stride(1),
        B, S, H,
        D_h,
        IS_FP16, IS_BF16,
    )

    return out


# ──────────────────────────────────────────────────────────────────────────────
# replacement_func: returns the callable (not a call to it)
# ──────────────────────────────────────────────────────────────────────────────
def replacement_func():
    return mha_triton_wrapper