import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def _flash_attn_fwd(
    Q, K, V, Out,
    # Q strides: [B, H, M, D]
    sq_b, sq_h, sq_m, sq_d,
    # K strides: [B, H, D, N]  (K is already transposed)
    sk_b, sk_h, sk_d, sk_n,
    # V strides: [B, H, N, D]
    sv_b, sv_h, sv_n, sv_d,
    H, M, N,
    HEAD_DIM: tl.constexpr,      # actual = 80
    HEAD_DIM_PAD: tl.constexpr,  # padded to nearest power-of-2 = 128
    DTYPE_ID: tl.constexpr,      # 0=fp32, 1=fp16, 2=bf16
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    m_block = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H

    offs_m = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PAD)
    d_mask = offs_d < HEAD_DIM
    m_mask = offs_m < M

    q_base = Q + b * sq_b + h * sq_h
    k_base = K + b * sk_b + h * sk_h
    v_base = V + b * sv_b + h * sv_h

    # Load Q block: [BLOCK_M, HEAD_DIM_PAD] as float32
    q = tl.load(
        q_base + offs_m[:, None] * sq_m + offs_d[None, :] * sq_d,
        mask=m_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Flash attention accumulators
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PAD], dtype=tl.float32)
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Loop over key/value blocks
    for n_start in range(0, N, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        n_mask = offs_n < N

        # Load K block: [HEAD_DIM_PAD, BLOCK_N]
        k = tl.load(
            k_base + offs_d[:, None] * sk_d + offs_n[None, :] * sk_n,
            mask=d_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, k)
        # Mask invalid key positions
        scores = tl.where(n_mask[None, :], scores, -1e9)

        # Online softmax: update m and l
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        exp_s = tl.exp(scores - m_new[:, None])
        l_i = alpha * l_i + tl.sum(exp_s, axis=1)

        # Load V block: [BLOCK_N, HEAD_DIM_PAD]
        v = tl.load(
            v_base + offs_n[:, None] * sv_n + offs_d[None, :] * sv_d,
            mask=n_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        # Accumulate weighted values
        acc = alpha[:, None] * acc + tl.dot(exp_s, v)
        m_i = m_new

    # Normalize by sum of exp weights
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    # Write output directly in transposed+reshaped format: [B, M, H*HEAD_DIM]
    # This fuses transpose(1,2) + reshape(1,257,-1)
    out_base = Out + b * (M * H * HEAD_DIM)
    out_off = offs_m[:, None] * (H * HEAD_DIM) + h * HEAD_DIM + offs_d[None, :]
    out_mask = m_mask[:, None] & d_mask[None, :]

    if DTYPE_ID == 1:  # float16
        tl.store(out_base + out_off, acc.to(tl.float16), mask=out_mask)
    elif DTYPE_ID == 2:  # bfloat16
        tl.store(out_base + out_off, acc.to(tl.bfloat16), mask=out_mask)
    else:  # float32
        tl.store(out_base + out_off, acc, mask=out_mask)


@torch.fx.wrap
def fused_attn_bf16(q, k_t, v):
    """Flash attention for bfloat16 inputs. Returns [B, M, H*D]."""
    B, H, M, D = q.shape
    N = k_t.shape[3]
    out = torch.empty(B, M, H * D, dtype=torch.bfloat16, device=q.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), B * H)
    _flash_attn_fwd[grid](
        q, k_t, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        H, M, N,
        HEAD_DIM=D,
        HEAD_DIM_PAD=128,
        DTYPE_ID=2,
    )
    return out


@torch.fx.wrap
def fused_attn_fp16(q, k_t, v):
    """Flash attention for float16 inputs. Returns [B, M, H*D]."""
    B, H, M, D = q.shape
    N = k_t.shape[3]
    out = torch.empty(B, M, H * D, dtype=torch.float16, device=q.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), B * H)
    _flash_attn_fwd[grid](
        q, k_t, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        H, M, N,
        HEAD_DIM=D,
        HEAD_DIM_PAD=128,
        DTYPE_ID=1,
    )
    return out


@torch.fx.wrap
def fused_attn_fp32(q, k_t, v):
    """Flash attention for float32 inputs. Returns [B, M, H*D]."""
    B, H, M, D = q.shape
    N = k_t.shape[3]
    out = torch.empty(B, M, H * D, dtype=torch.float32, device=q.device)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), B * H)
    _flash_attn_fwd[grid](
        q, k_t, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_t.stride(0), k_t.stride(1), k_t.stride(2), k_t.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        H, M, N,
        HEAD_DIM=D,
        HEAD_DIM_PAD=128,
        DTYPE_ID=0,
    )
    return out