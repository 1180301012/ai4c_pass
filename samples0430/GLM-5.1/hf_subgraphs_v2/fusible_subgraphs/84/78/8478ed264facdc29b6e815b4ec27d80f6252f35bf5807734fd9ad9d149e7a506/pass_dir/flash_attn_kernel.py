import torch
import triton
import triton.language as tl


@triton.jit
def flash_attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    scale_recip,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kk, stride_kn,
    stride_vb, stride_vh, stride_vm, stride_vk,
    stride_ob, stride_on, stride_oh, stride_od,
    Nq, Nk, D,
    n_heads,
    DTYPE_flag: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Flash Attention forward kernel.
    Computes: O = softmax(Q @ K^T / scale) @ V in permuted layout [B, Nq, H, D].
    Uses online softmax to avoid materializing the full attention matrix.
    """
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_h = off_hb % n_heads
    off_b = off_hb // n_heads

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    m_mask = offs_m < Nq
    d_mask = offs_d < D

    q_base = off_b * stride_qb + off_h * stride_qh
    k_base = off_b * stride_kb + off_h * stride_kh
    v_base = off_b * stride_vb + off_h * stride_vh

    # Load Q block: [BLOCK_M, BLOCK_D]
    q_ptrs = Q_ptr + q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q = tl.load(q_ptrs, mask=m_mask[:, None] & d_mask[None, :], other=0.0)

    # Initialize online softmax accumulators
    # Use -1e30 instead of -inf to avoid NaN from -inf - (-inf)
    m_i = tl.full([BLOCK_M], -1e30, dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    # Iterate over K/V blocks
    for start_n in range(0, Nk, BLOCK_N):
        offs_n_curr = start_n + offs_n
        n_curr_mask = offs_n_curr < Nk

        # Load K^T block: [BLOCK_D, BLOCK_N]
        # K^T has shape [B, H, D, Nk], stored as (d, n) with strides (stride_kk, stride_kn)
        k_ptrs = K_ptr + k_base + offs_d[:, None] * stride_kk + offs_n_curr[None, :] * stride_kn
        k = tl.load(k_ptrs, mask=d_mask[:, None] & n_curr_mask[None, :], other=0.0)

        # Load V block: [BLOCK_N, BLOCK_D]
        # V has shape [B, H, Nk, D], stored as (n, d) with strides (stride_vm, stride_vk)
        v_ptrs = V_ptr + v_base + offs_n_curr[:, None] * stride_vm + offs_d[None, :] * stride_vk
        v = tl.load(v_ptrs, mask=n_curr_mask[:, None] & d_mask[None, :], other=0.0)

        # Compute QK^T: [BLOCK_M, BLOCK_D] @ [BLOCK_D, BLOCK_N] = [BLOCK_M, BLOCK_N]
        # For fp16/bf16, tl.dot uses tensor cores and accumulates in fp32
        s = tl.dot(q, k, allow_tf32=DTYPE_flag != 0)
        s = s.to(tl.float32) * scale_recip

        # Mask invalid positions in attention scores
        # Invalid query rows (beyond Nq) and invalid key columns (beyond Nk) get -1e30
        valid_mask = m_mask[:, None] & n_curr_mask[None, :]
        s = tl.where(valid_mask, s, -1e30)

        # ---- Online softmax update ----
        # Compute row-wise max of current block
        m_ij = tl.max(s, axis=1)  # [BLOCK_M]
        m_new = tl.maximum(m_i, m_ij)  # [BLOCK_M] - running max update

        # Correction factor for previous accumulator
        alpha = tl.exp(m_i - m_new)  # [BLOCK_M]

        # Rescale previous accumulator and running sum
        acc = acc * alpha[:, None]
        l_i = l_i * alpha

        # Compute normalized probabilities for current block
        p = tl.exp(s - m_new[:, None])  # [BLOCK_M, BLOCK_N]
        # Mask invalid probabilities to 0
        p = tl.where(valid_mask, p, 0.0)

        # Compute PV contribution: [BLOCK_M, BLOCK_N] @ [BLOCK_N, BLOCK_D] = [BLOCK_M, BLOCK_D]
        # Cast p to input dtype for tensor core usage (fp16/bf16)
        if DTYPE_flag == 0:  # float32
            pv = tl.dot(p, v, allow_tf32=False)
        elif DTYPE_flag == 1:  # float16
            pv = tl.dot(p.to(tl.float16), v, allow_tf32=True)
        else:  # bfloat16 (DTYPE_flag == 2)
            pv = tl.dot(p.to(tl.bfloat16), v, allow_tf32=True)

        acc += pv.to(tl.float32)

        # Update running sum
        l_ij = tl.sum(p, axis=1)  # [BLOCK_M]
        l_i += l_ij

        # Update running max
        m_i = m_new

    # ---- Finalize: normalize by running sum ----
    # Avoid division by zero for invalid rows (l_i == 0)
    l_i_safe = tl.where(l_i > 1e-10, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    # Zero out invalid query rows and invalid dimension elements
    acc = tl.where(m_mask[:, None], acc, 0.0)
    acc = acc * d_mask[None, :].to(tl.float32)

    # Store output in permuted layout: [B, Nq, H, D]
    # Strides: stride_ob (batch), stride_on (Nq), stride_oh (heads), stride_od (dim)
    o_ptrs = O_ptr + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_on + offs_d[None, :] * stride_od

    store_mask = m_mask[:, None] & d_mask[None, :]
    if DTYPE_flag == 0:  # float32
        tl.store(o_ptrs, acc, mask=store_mask)
    elif DTYPE_flag == 1:  # float16
        tl.store(o_ptrs, acc.to(tl.float16), mask=store_mask)
    else:  # bfloat16
        tl.store(o_ptrs, acc.to(tl.bfloat16), mask=store_mask)


# Scale values mapped from route strings
SCALE_MAP = {
    "route_scale5657_drop0": 5.656854249492381,
    "route_scale5657_drop01": 5.656854249492381,
    "route_scale8_drop0": 8.0,
    "route_scale8_drop01": 8.0,
    "route_scale6_drop0": 6.0,
    "route_scale6_drop01": 6.0,
    "route_scale6928_drop0": 6.928203230275509,
    "route_scale6928_drop01": 6.928203230275509,
}


@torch.fx.wrap
def flash_attn_dispatch(q, k_t, v, route):
    """
    Dispatch wrapper for flash attention.
    Since dropout with training=False is identity, all routes compute the same thing.
    The route string determines the scale factor.
    
    q: [B, H, Nq, D] - query tensor
    k_t: [B, H, D, Nk] - key tensor (already transposed)
    v: [B, H, Nk, D] - value tensor
    route: string - route identifier for scale factor
    Returns: [B, Nq, H, D] - output in permuted+contiguous layout
    """
    scale = SCALE_MAP[route]
    B, H, Nq, D = q.shape
    Nk = k_t.shape[-1]

    # Allocate output in permuted+contiguous layout: [B, Nq, H, D]
    o = torch.empty((B, Nq, H, D), dtype=q.dtype, device=q.device)

    # Precompute reciprocal for faster multiplication (vs division)
    scale_recip = 1.0 / scale

    # Get strides for all tensors
    stride_qb, stride_qh, stride_qm, stride_qk = q.stride()
    stride_kb, stride_kh, stride_kk, stride_kn = k_t.stride()
    stride_vb, stride_vh, stride_vm, stride_vk = v.stride()
    stride_ob, stride_on, stride_oh, stride_od = o.stride()

    # Determine dtype flag for kernel
    if q.dtype == torch.float32:
        DTYPE_flag = 0
    elif q.dtype == torch.float16:
        DTYPE_flag = 1
    else:  # bfloat16
        DTYPE_flag = 2

    # Compute BLOCK_D: next power of 2 >= D, minimum 16 for tl.dot
    BLOCK_D = 1
    while BLOCK_D < max(D, 16):
        BLOCK_D *= 2

    # Block sizes for tiling
    BLOCK_M = 64
    BLOCK_N = 64

    # Grid: (num_query_blocks, batch * heads)
    grid = (triton.cdiv(Nq, BLOCK_M), B * H)

    flash_attn_fwd_kernel[grid](
        q, k_t, v, o,
        scale_recip,
        stride_qb, stride_qh, stride_qm, stride_qk,
        stride_kb, stride_kh, stride_kk, stride_kn,
        stride_vb, stride_vh, stride_vm, stride_vk,
        stride_ob, stride_on, stride_oh, stride_od,
        Nq, Nk, D,
        H,
        DTYPE_flag=DTYPE_flag,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return o