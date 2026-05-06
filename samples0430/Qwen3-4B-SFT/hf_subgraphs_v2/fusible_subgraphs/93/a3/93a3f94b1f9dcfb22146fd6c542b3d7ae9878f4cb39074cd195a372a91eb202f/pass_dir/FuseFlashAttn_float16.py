import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_D': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32,  'BLOCK_D': 128}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32,  'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32,  'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_D': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_D': 128}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'H', 'K'],
)
@triton.jit
def _flash_attn_fwd_f16(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kd, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_om, stride_on,
    B, H, M, N_total, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    DTYPE_CODE: tl.constexpr,
):
    offs_bh = tl.program_id(0)
    offs_m  = tl.program_id(1) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d  = tl.arange(0, BLOCK_D)

    q_col = offs_d % K
    bh    = offs_d // K

    mask_m = offs_m < M

    # Load Q block [BLOCK_M, BLOCK_D]
    q_offs = (offs_bh * stride_qh + offs_m[:, None] * stride_qm + q_col[None, :] * stride_qk)
    q = tl.load(Q_ptr + q_offs, mask=mask_m[:, None], other=0.0).to(tl.float32)

    # Online-softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for j in range(0, tl.cdiv(N_total, BLOCK_N)):
        offs_n = j * BLOCK_N + tl.arange(0, BLOCK_N)
        mask_n = offs_n < N_total

        # Load K.T [BLOCK_D, BLOCK_N]  (k is pre-transposed)
        k_offs = (offs_kh * stride_kh + offs_d[:, None] * stride_kd + offs_n[None, :] * stride_kn)
        k = tl.load(K_ptr + k_offs, mask=mask_n[None, :], other=float('-inf')).to(tl.float32)

        # FP32 attention scores [BLOCK_M, BLOCK_N]
        qk = tl.dot(q.to(tl.float32), k).to(tl.float32)
        qk = tl.where(mask_n[None, :], qk, float('-inf'))
        # Out-of-bounds query rows → -inf
        qk = tl.where(mask_m[:, None] & mask_n[None, :], qk, float('-inf'))

        # Online softmax
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        p     = tl.exp(qk - m_new[:, None])
        alpha = tl.exp(m_i - m_new)

        l_i  = alpha * l_i + tl.sum(p, axis=1)
        acc  = acc * alpha[:, None]

        # Load V [BLOCK_N, BLOCK_D]
        v_offs = (offs_vh * stride_vh + offs_n[:, None] * stride_vn + q_col[None, :] * stride_vk)
        v = tl.load(V_ptr + v_offs, mask=mask_n[:, None], other=0.0).to(tl.float32)

        acc  += tl.dot(p, v)
        m_i   = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store, with dtype conversion
    out_offs = (offs_ob * stride_ob + offs_m[:, None] * stride_om + offs_on[None, :] * stride_on)
    if DTYPE_CODE == 0:
        tl.store(Out_ptr + out_offs, acc.to(tl.bfloat16),  mask=mask_m[:, None])
    elif DTYPE_CODE == 1:
        tl.store(Out_ptr + out_offs, acc.to(tl.float16),   mask=mask_m[:, None])
    else:
        tl.store(Out_ptr + out_offs, acc,                  mask=mask_m[:, None])


@torch.fx.wrap
def launch_flash_attn_f16(q, k, v):
    B, H, M, K = q.shape
    N      = k.shape[3]
    DTYPE_CODE = 1  # float16
    Out    = torch.empty(B, M, H * K, device=q, dtype=torch.float16)
    grid   = (H * B, triton.cdiv(M, 16))
    _flash_attn_fwd_f16[grid](
        q, k, v, Out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2),
        B, H, M, N, K,
        DTYPE_CODE=DTYPE_CODE,
    )
    return Out


def pattern(in_0, in_1, in_2):
    matmul   = torch.matmul(in_0, in_1)
    tmp_1    = matmul * 1.0
    tmp_2    = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3    = tmp_2.to(torch.float32)
    tmp_4    = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    to       = tmp_4.to(torch.float16)
    matmul_1 = torch.matmul(to, in_2)
    tmp_6    = matmul_1.transpose(1, 2)
    tmp_7    = tmp_6.contiguous()
    tmp_8    = tmp_7.reshape(1, 257, -1)
    tmp_9    = tmp_8.contiguous()
    return (tmp_9,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return launch_flash_attn_f16