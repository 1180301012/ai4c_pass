import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
    ],
    key=['N_CTX', 'HEAD_DIM', 'HEAD_DIM_PADDED', 'IS_BF16', 'IS_F16'],
)
@triton.jit
def _flash_attn_fwd(
    Q_ptr, KT_ptr, V_ptr, Out_ptr,
    # Q strides [B, H, N, D]
    stride_qb, stride_qh, stride_qn, stride_qd,
    # KT strides [B, H, D, N]
    stride_kb, stride_kh, stride_kd, stride_kn,
    # V strides [B, H, N, D]
    stride_vb, stride_vh, stride_vn, stride_vd,
    # Out strides [B, N, H*D]
    stride_ob, stride_on,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,          # actual head dim (e.g. 80)
    HEAD_DIM_PADDED: tl.constexpr,   # next power of 2 >= HEAD_DIM (e.g. 128)
    H: tl.constexpr,
    IS_BF16: tl.constexpr,
    IS_F16: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_m = tl.program_id(2)

    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)   # padded to power of 2
    d_mask = offs_d < HEAD_DIM               # valid head-dim lanes
    q_mask = offs_m < N_CTX

    # Load Q [BLOCK_M, HEAD_DIM_PADDED], zero out padded lanes
    q_ptrs = (Q_ptr
              + pid_b * stride_qb
              + pid_h * stride_qh
              + offs_m[:, None] * stride_qn
              + offs_d[None, :] * stride_qd)
    q = tl.load(q_ptrs,
                mask=(q_mask[:, None] & d_mask[None, :]),
                other=0.0).to(tl.float32)

    # Running stats for online softmax
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    n_blocks = tl.cdiv(N_CTX, BLOCK_N)
    for j in range(0, n_blocks):
        start_n = j * BLOCK_N
        offs_n = start_n + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < N_CTX

        # Load KT [HEAD_DIM_PADDED, BLOCK_N], zero padded D rows
        kt_ptrs = (KT_ptr
                   + pid_b * stride_kb
                   + pid_h * stride_kh
                   + offs_d[:, None] * stride_kd
                   + offs_n[None, :] * stride_kn)
        kt = tl.load(kt_ptrs,
                     mask=(d_mask[:, None] & kv_mask[None, :]),
                     other=0.0).to(tl.float32)

        # Attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, kt)
        scores = tl.where(kv_mask[None, :], scores, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        scores_exp = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(scores_exp, axis=1)
        acc = acc * alpha[:, None]

        # Load V [BLOCK_N, HEAD_DIM_PADDED], zero padded D cols
        v_ptrs = (V_ptr
                  + pid_b * stride_vb
                  + pid_h * stride_vh
                  + offs_n[:, None] * stride_vn
                  + offs_d[None, :] * stride_vd)
        v = tl.load(v_ptrs,
                    mask=(kv_mask[:, None] & d_mask[None, :]),
                    other=0.0).to(tl.float32)

        acc += tl.dot(scores_exp, v)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Cast output
    if IS_BF16:
        out_val = acc.to(tl.bfloat16)
    elif IS_F16:
        out_val = acc.to(tl.float16)
    else:
        out_val = acc

    # Write to [B, N, H*D] directly (fuses transpose(1,2) + reshape)
    out_ptrs = (Out_ptr
                + pid_b * stride_ob
                + offs_m[:, None] * stride_on
                + (pid_h * HEAD_DIM + offs_d[None, :]))
    tl.store(out_ptrs, out_val,
             mask=(q_mask[:, None] & d_mask[None, :]))


@torch.fx.wrap
def flash_attn_forward(in_0, in_1, in_2):
    """
    in_0: Q  [B, H, N, D]
    in_1: KT [B, H, D, N]  (key already transposed)
    in_2: V  [B, H, N, D]
    Returns: [B, N, H*D]  (fuses softmax + matmul + transpose + reshape)
    """
    B, H, N, D = in_0.shape
    dtype = in_0.dtype

    is_bf16 = (dtype == torch.bfloat16)
    is_f16 = (dtype == torch.float16)

    HD_PAD = triton.next_power_of_2(D)   # 128 for D=80

    out = torch.empty((B, N, H * D), dtype=dtype, device=in_0.device)

    stride_ob = N * H * D
    stride_on = H * D

    grid = lambda meta: (B, H, triton.cdiv(N, meta['BLOCK_M']))

    _flash_attn_fwd[grid](
        in_0, in_1, in_2, out,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        stride_ob, stride_on,
        N_CTX=N,
        HEAD_DIM=D,
        HEAD_DIM_PADDED=HD_PAD,
        H=H,
        IS_BF16=is_bf16,
        IS_F16=is_f16,
    )
    return out