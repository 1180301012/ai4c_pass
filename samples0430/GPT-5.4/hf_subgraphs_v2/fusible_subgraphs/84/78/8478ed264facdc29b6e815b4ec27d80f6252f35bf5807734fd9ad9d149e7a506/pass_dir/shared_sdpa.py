import torch
import triton
import triton.language as tl

_LOG2E = 1.4426950408889634


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_D": 64}, num_warps=1, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_D": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 32, "BLOCK_D": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_D": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_D": 64}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "D"],
)
@triton.jit
def _sdpa_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    o_ptr,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_qd,
    stride_kb,
    stride_kh,
    stride_kd,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_vd,
    stride_ob,
    stride_om,
    stride_oh,
    stride_od,
    H,
    M,
    N,
    D,
    sm_scale_log2,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_D)

    valid_m = offs_m < M
    valid_d = offs_d < D

    q_ptrs = (
        q_ptr
        + b * stride_qb
        + h * stride_qh
        + offs_m[:, None] * stride_qm
        + offs_d[None, :] * stride_qd
    )
    q = tl.load(q_ptrs, mask=valid_m[:, None] & valid_d[None, :], other=0.0)

    m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    l_i = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

    for start_n in range(0, N, BLOCK_N):
        curr_n = start_n + offs_n
        valid_n = curr_n < N

        k_ptrs = (
            k_ptr
            + b * stride_kb
            + h * stride_kh
            + offs_d[:, None] * stride_kd
            + curr_n[None, :] * stride_kn
        )
        k = tl.load(k_ptrs, mask=valid_d[:, None] & valid_n[None, :], other=0.0)

        qk = tl.dot(q, k)
        qk = qk * sm_scale_log2
        qk = tl.where(valid_m[:, None] & valid_n[None, :], qk, -float("inf"))

        m_ij = tl.max(qk, axis=1)
        m_ij = tl.where(valid_m, m_ij, 0.0)
        m_new = tl.maximum(m_i, m_ij)

        alpha = tl.math.exp2(m_i - m_new)
        alpha = tl.where(valid_m, alpha, 1.0)

        p = tl.math.exp2(qk - m_new[:, None])
        p = tl.where(valid_m[:, None] & valid_n[None, :], p, 0.0)

        v_ptrs = (
            v_ptr
            + b * stride_vb
            + h * stride_vh
            + curr_n[:, None] * stride_vn
            + offs_d[None, :] * stride_vd
        )
        v = tl.load(v_ptrs, mask=valid_n[:, None] & valid_d[None, :], other=0.0)

        acc = acc * alpha[:, None] + tl.dot(p, v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    l_i_safe = tl.where(valid_m, l_i, 1.0)
    acc = acc / l_i_safe[:, None]

    o_ptrs = (
        o_ptr
        + b * stride_ob
        + offs_m[:, None] * stride_om
        + h * stride_oh
        + offs_d[None, :] * stride_od
    )
    tl.store(o_ptrs, acc, mask=valid_m[:, None] & valid_d[None, :])


@torch.fx.wrap
def sdpa_wrapper(q, k, v, scale, dropout_p):
    batch = q.shape[0]
    heads = q.shape[1]
    m_size = q.shape[2]
    d_size = q.shape[3]
    n_size = v.shape[2]

    out_4d = torch.empty((batch, m_size, heads, d_size), device=q.device, dtype=q.dtype)
    if batch == 0 or heads == 0 or m_size == 0 or n_size == 0 or d_size == 0:
        return out_4d

    sm_scale_log2 = (_LOG2E / float(scale))
    grid = lambda META: (triton.cdiv(m_size, META["BLOCK_M"]), batch * heads)

    _sdpa_forward_kernel[grid](
        q,
        k,
        v,
        out_4d,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        out_4d.stride(0),
        out_4d.stride(1),
        out_4d.stride(2),
        out_4d.stride(3),
        heads,
        m_size,
        n_size,
        d_size,
        sm_scale_log2,
    )
    return out_4d.permute(0, 2, 1, 3).contiguous().view(batch, m_size, heads * d_size)