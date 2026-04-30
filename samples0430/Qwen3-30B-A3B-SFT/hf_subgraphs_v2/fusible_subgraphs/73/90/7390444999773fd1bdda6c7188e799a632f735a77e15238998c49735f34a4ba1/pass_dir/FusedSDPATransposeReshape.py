import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: scaled_dot_product_attention + transpose(1,2)
# ---------------------------------------------------------------------------

def pattern(query, key, value, attn_mask):
    sdpa = torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False
    )
    t = sdpa.transpose(1, 2)
    return t


def replacement_args(query, key, value, attn_mask):
    return (query, key, value, attn_mask)


# ---------------------------------------------------------------------------
# Triton Flash Attention kernel (v3)
# Output layout: [B, S, H, D]  — same layout as model's downstream transpose
#   result, so reshape is a VIEW with zero extra memory copies.
#
# Key optimisations:
#   • num_stages=2 → Triton software-pipelines loads of tile k+1 with
#     computation on tile k, hiding memory latency
#   • Block-pointer API for K, V, mask, output (coalesced 2D-tile loads)
#   • No @triton.autotune (avoids per-call overhead for re-dispatch)
# ---------------------------------------------------------------------------

@triton.jit
def _fused_sdpa_kernel(
    Q_ptr, K_ptr, V_ptr, Mask_ptr, Out_ptr,
    # Q strides [B, H, S, D] (4-D; head dim present)
    stride_qb, stride_qh, stride_qs, stride_qd,
    # K strides [B, H, S, D]
    stride_kb, stride_kh, stride_ks, stride_kd,
    # V strides [B, H, S, D]
    stride_vb, stride_vh, stride_vs, stride_vd,
    # Mask strides [B, 1, S, S]  → head dim skipped (stride = S*S, not used)
    stride_mb, stride_ms, stride_mt,
    # Out strides [B, S, H, D]
    stride_ob, stride_os, stride_oh, stride_od,
    # Tensor dims
    B, H, S, D,
    SM, TM,
    # Softmax scale factor
    scale,
    # Compile-time dtype flags
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    # Tile sizes (constexpr for tl.arange / tl.dot)
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_s  = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh  % H

    s_start = pid_s * BLOCK_S
    s_offs  = s_start + tl.arange(0, BLOCK_S)
    d_offs  = tl.arange(0, BLOCK_D)
    s_mask  = s_offs < S

    # Base pointer for Q (4-D layout [B, H, S, D])
    q_base = Q_ptr + b * stride_qb + h * stride_qh

    # Online-softmax accumulators (float32 for numerical stability)
    m_i = tl.full([BLOCK_S], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_S],              dtype=tl.float32)
    acc = tl.zeros([BLOCK_S, BLOCK_D],     dtype=tl.float32)

    k_base = K_ptr + b * stride_kb + h * stride_kh
    v_base = V_ptr + b * stride_vb + h * stride_vh
    m_base = Mask_ptr + b * stride_mb   # head dim == 1 → skip it

    # Load Q ONCE before the inner loop (avoids redundant reload each KV tile)
    q_ptrs = q_base + s_offs[:, None] * stride_qs + d_offs[None, :] * stride_qd
    q_raw = tl.load(q_ptrs, mask=s_mask[:, None], other=0.0)
    if IS_FP16:
        q_raw = q_raw.to(tl.float16)
    elif IS_BF16:
        q_raw = q_raw.to(tl.bfloat16)

    # Main loop over KV tiles  (num_stages=2 pipelines load of tile k+1
    # with compute on tile k, hiding memory round-trip latency)
    for t_start in range(0, S, BLOCK_S):
        t_offs = t_start + tl.arange(0, BLOCK_S)
        t_mask = t_offs < S

        # --- K tile [BLOCK_S, BLOCK_D] ----------------------------------------
        k_ptrs = k_base + t_offs[:, None] * stride_ks + d_offs[None, :] * stride_kd
        k = tl.load(k_ptrs, mask=t_mask[:, None], other=0.0)
        if IS_FP16:
            k = k.to(tl.float16)
        elif IS_BF16:
            k = k.to(tl.bfloat16)

        # --- QK^T [BLOCK_S, BLOCK_S] ------------------------------------------
        # q_raw is already in the target dtype (loaded once, kept in registers)
        qk = tl.dot(q_raw, tl.trans(k), out_dtype=tl.float32) * scale

        # --- Add additive attention mask [BLOCK_S, BLOCK_S] --------------------
        sm_ptrs = m_base + s_offs[:, None] * stride_ms + t_offs[None, :] * stride_mt
        sm_mask = s_mask[:, None] & t_mask[None, :]
        mask_val = tl.load(sm_ptrs, mask=sm_mask, other=0.0).to(tl.float32)
        qk = qk + mask_val
        qk = tl.where(sm_mask, qk, float('-inf'))

        # --- Online softmax update --------------------------------------------
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(qk - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        acc   = acc * alpha[:, None]

        # --- V tile [BLOCK_S, BLOCK_D] ----------------------------------------
        v_ptrs = v_base + t_offs[:, None] * stride_vs + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=t_mask[:, None], other=0.0)
        if IS_FP16:
            v = v.to(tl.float16)
        elif IS_BF16:
            v = v.to(tl.bfloat16)

        acc  += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i  = m_new

    # Normalise
    acc = acc / l_i[:, None]

    # Store output in [B, S, H, D] layout  (downstream transpose(1,2) is a VIEW)
    o_base = Out_ptr + b * stride_ob + h * stride_oh
    o_ptrs = o_base + s_offs[:, None] * stride_os + d_offs[None, :] * stride_od
    if IS_FP16:
        tl.store(o_ptrs, acc.to(tl.float16), mask=s_mask[:, None])
    elif IS_BF16:
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=s_mask[:, None])
    else:
        tl.store(o_ptrs, acc, mask=s_mask[:, None])


# ---------------------------------------------------------------------------
# Kernel wrapper  (@torch.fx.wrap required)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_sdpa_wrapper(query, key, value, attn_mask):
    """
    Output in [B, S, H, D] — the same memory layout the model's downstream
    transpose(1,2) would produce.  Therefore:
      • transpose(1,2) is a stride-only VIEW  (no data copy)
      • reshape(B, S, H*D) is also a VIEW         (last 2 dims contiguous)
    """
    q0, q1, q2, q3 = query.shape
    a2, a3            = attn_mask.shape[2], attn_mask.shape[3]

    fp16 = query.dtype == torch.float16
    bf16 = query.dtype == torch.bfloat16

    out = torch.empty((q0, q2, q1, q3), dtype=query.dtype, device=query.device)

    BS = 64
    grid = (q0 * q1, (q2 + BS - 1) // BS)

    _fused_sdpa_kernel[grid](
        query, key, value, attn_mask, out,
        query.stride(0), query.stride(1), query.stride(2), query.stride(3),
        key.stride(0),   key.stride(1),   key.stride(2),   key.stride(3),
        value.stride(0), value.stride(1), value.stride(2), value.stride(3),
        attn_mask.stride(0), attn_mask.stride(2), attn_mask.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        q0, q1, q2, q3,
        a2, a3,
        1.0 / (q3 ** 0.5),
        fp16, bf16,
        BS, 64,
        num_stages=2,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func: zero-arg, returns the callable (not a call result)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_sdpa_wrapper