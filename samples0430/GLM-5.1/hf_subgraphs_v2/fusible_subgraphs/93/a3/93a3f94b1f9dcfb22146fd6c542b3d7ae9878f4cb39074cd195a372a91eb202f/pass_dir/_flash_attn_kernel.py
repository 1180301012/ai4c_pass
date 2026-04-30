import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_kernel(
    Q_ptr, K_T_ptr, V_ptr, O_ptr,
    seq_len, head_dim,
    stride_qh, stride_qs, stride_qd,
    stride_kh, stride_kd, stride_ks,
    stride_vh, stride_vs, stride_vd,
    stride_os, stride_ohd,
    BLOCK_SQ: tl.constexpr,
    BLOCK_SK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
    OUTPUT_FP16: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_sq = tl.program_id(1)

    sq_start = pid_sq * BLOCK_SQ
    sq_offsets = sq_start + tl.arange(0, BLOCK_SQ)
    d_offsets = tl.arange(0, BLOCK_D)

    sq_mask = sq_offsets < seq_len
    d_mask = d_offsets < head_dim

    # Load Q block: [BLOCK_SQ, BLOCK_D] in input dtype
    q_ptrs = Q_ptr + pid_h * stride_qh + sq_offsets[:, None] * stride_qs + d_offsets[None, :] * stride_qd
    Q_block = tl.load(q_ptrs, mask=sq_mask[:, None] & d_mask[None, :], other=0.0)

    # Initialize accumulators (float32)
    # Use -1e30 instead of -inf to avoid NaN in exp(-inf - (-inf))
    m_i = tl.full([BLOCK_SQ], -1e30, tl.float32)
    l_i = tl.zeros([BLOCK_SQ], tl.float32)
    O_acc = tl.zeros([BLOCK_SQ, BLOCK_D], tl.float32)

    # Iterate over K/V blocks
    for sk_start in range(0, seq_len, BLOCK_SK):
        sk_offsets = sk_start + tl.arange(0, BLOCK_SK)
        sk_mask = sk_offsets < seq_len

        # Load K_T block: [BLOCK_D, BLOCK_SK] in input dtype
        k_ptrs = K_T_ptr + pid_h * stride_kh + d_offsets[:, None] * stride_kd + sk_offsets[None, :] * stride_ks
        K_block = tl.load(k_ptrs, mask=d_mask[:, None] & sk_mask[None, :], other=0.0)

        # Compute S = Q @ K_T: [BLOCK_SQ, BLOCK_SK]
        S = tl.dot(Q_block, K_block)

        # Mask out-of-bounds positions to -inf for correct softmax
        S = tl.where(sq_mask[:, None] & sk_mask[None, :], S, float('-inf'))

        # Online softmax update
        m_ij = tl.max(S, axis=1)  # [BLOCK_SQ]
        m_new = tl.maximum(m_i, m_ij)  # [BLOCK_SQ]

        # Correction factor for previous accumulations
        alpha = tl.exp(m_i - m_new)  # [BLOCK_SQ]
        l_i *= alpha
        O_acc *= alpha[:, None]

        # New attention weights
        P = tl.exp(S - m_new[:, None])  # [BLOCK_SQ, BLOCK_SK]

        # Update l
        l_i += tl.sum(P, axis=1)  # [BLOCK_SQ]

        # Load V block: [BLOCK_SK, BLOCK_D] in input dtype
        v_ptrs = V_ptr + pid_h * stride_vh + sk_offsets[:, None] * stride_vs + d_offsets[None, :] * stride_vd
        V_block = tl.load(v_ptrs, mask=sk_mask[:, None] & d_mask[None, :], other=0.0)

        # Cast P to output dtype for P @ V dot product (matching original cast after dropout)
        if OUTPUT_BF16:
            P_cast = P.to(tl.bfloat16)
        elif OUTPUT_FP16:
            P_cast = P.to(tl.float16)
        else:
            P_cast = P

        # Accumulate: O += P @ V
        O_acc += tl.dot(P_cast, V_block)

        # Update m
        m_i = m_new

    # Finalize: O = O_acc / l_i
    # Handle l_i = 0 for out-of-bounds sq positions
    l_i_safe = tl.where(l_i == 0, 1.0, l_i)
    O_fp32 = O_acc / l_i_safe[:, None]
    O_fp32 = tl.where(sq_mask[:, None], O_fp32, 0.0)

    # Cast to output dtype
    if OUTPUT_BF16:
        O_store = O_fp32.to(tl.bfloat16)
    elif OUTPUT_FP16:
        O_store = O_fp32.to(tl.float16)
    else:
        O_store = O_fp32

    # Store output in [batch, seq_q, heads * head_dim] layout
    # For head h at position (sq, d): offset = sq * stride_os + h * head_dim + d * stride_ohd
    o_ptrs = O_ptr + sq_offsets[:, None] * stride_os + pid_h * head_dim + d_offsets[None, :] * stride_ohd
    tl.store(o_ptrs, O_store, mask=sq_mask[:, None] & d_mask[None, :])


@torch.fx.wrap
def flash_attn_dispatch(Q, K_T, V, route):
    batch = Q.shape[0]
    heads = Q.shape[1]
    seq_len = Q.shape[2]
    head_dim = Q.shape[3]

    if route == "bf16":
        output_dtype = torch.bfloat16
        OUTPUT_BF16 = True
        OUTPUT_FP16 = False
    elif route == "fp16":
        output_dtype = torch.float16
        OUTPUT_BF16 = False
        OUTPUT_FP16 = True
    else:  # fp32
        output_dtype = torch.float32
        OUTPUT_BF16 = False
        OUTPUT_FP16 = False

    # Allocate output: [batch, seq_len, heads * head_dim]
    O = torch.empty(batch, seq_len, heads * head_dim, dtype=output_dtype, device=Q.device)

    BLOCK_SQ = 32
    BLOCK_SK = 32
    BLOCK_D = head_dim

    grid = (heads, triton.cdiv(seq_len, BLOCK_SQ))

    flash_attn_kernel[grid](
        Q, K_T, V, O,
        seq_len, head_dim,
        Q.stride(1), Q.stride(2), Q.stride(3),
        K_T.stride(1), K_T.stride(2), K_T.stride(3),
        V.stride(1), V.stride(2), V.stride(3),
        O.stride(1), O.stride(2),
        BLOCK_SQ=BLOCK_SQ,
        BLOCK_SK=BLOCK_SK,
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=OUTPUT_BF16,
        OUTPUT_FP16=OUTPUT_FP16,
    )

    return O