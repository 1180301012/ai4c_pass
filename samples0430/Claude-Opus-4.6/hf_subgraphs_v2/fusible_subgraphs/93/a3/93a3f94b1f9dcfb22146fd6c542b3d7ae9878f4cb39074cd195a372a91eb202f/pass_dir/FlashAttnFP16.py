import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul * 1.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3 = tmp_2.to(torch.float32)
    tmp_4 = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    to = tmp_4.to(torch.float16)
    matmul_1 = torch.matmul(to, in_2)
    tmp_6 = matmul_1.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def flash_attention_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qm, stride_qk,
    stride_kb, stride_kh, stride_kk, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_vk,
    stride_ob, stride_om, stride_ok,
    num_heads,
    seq_len: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // num_heads
    head_idx = pid_bh % num_heads

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)

    # Load Q [BLOCK_M, BLOCK_D]
    q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    q_ptrs = q_base + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    q_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Online softmax accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)

        # Load K [BLOCK_D, BLOCK_N]
        k_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
        k_ptrs = k_base + offs_d[:, None] * stride_kk + offs_n[None, :] * stride_kn
        k_mask = (offs_d[:, None] < head_dim) & (offs_n[None, :] < seq_len)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # QK [BLOCK_M, BLOCK_N]
        qk = tl.dot(q, k)
        qk = tl.where(offs_n[None, :] < seq_len, qk, float('-inf'))

        # Online softmax
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V [BLOCK_N, BLOCK_D]
        v_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh
        v_ptrs = v_base + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vk
        v_mask = (offs_n[:, None] < seq_len) & (offs_d[None, :] < head_dim)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        # PV accumulate
        acc += tl.dot(p.to(v.dtype), v)

        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store in [B, M, H*D] layout
    out_base = Out_ptr + batch_idx * stride_ob
    out_ptrs = out_base + offs_m[:, None] * stride_om + (head_idx * head_dim + offs_d[None, :]) * stride_ok
    out_mask = (offs_m[:, None] < seq_len) & (offs_d[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


@torch.fx.wrap
def flash_attention_fp16(in_0, in_1, in_2):
    B, H, M, D = in_0.shape
    out = torch.empty((B, M, H * D), dtype=in_0.dtype, device=in_0.device)

    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_D = 128

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    flash_attention_kernel[grid](
        in_0, in_1, in_2, out,
        in_0.stride(0), in_0.stride(1), in_0.stride(2), in_0.stride(3),
        in_1.stride(0), in_1.stride(1), in_1.stride(2), in_1.stride(3),
        in_2.stride(0), in_2.stride(1), in_2.stride(2), in_2.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        num_heads=H,
        seq_len=M,
        head_dim=D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )

    return out


def replacement_func():
    return flash_attention_fp16