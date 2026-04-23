import torch
import triton
import triton.language as tl


@triton.jit
def _fused_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d, stride_k_n,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_h, stride_o_m, stride_o_d,
    H, M, N,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    b = pid_bh // H
    h = pid_bh % H

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, D_HEAD)

    m_mask = offs_m < M

    q_ptrs = (
        q_ptr
        + b * stride_q_b
        + h * stride_q_h
        + offs_m[:, None] * stride_q_m
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    max_logits = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denom = tl.zeros((BLOCK_M,), tl.float32)
    acc = tl.zeros((BLOCK_M, D_HEAD), tl.float32)

    n_start = 0
    while n_start < N:
        n_curr = n_start + offs_n
        n_mask = n_curr < N

        k_ptrs = (
            k_ptr
            + b * stride_k_b
            + h * stride_k_h
            + offs_d[:, None] * stride_k_d
            + n_curr[None, :] * stride_k_n
        )
        k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

        logits = tl.dot(q, k) * scale
        logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, -float("inf"))

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_logits, block_max)
        alpha = tl.exp(max_logits - new_max)
        probs = tl.exp(logits - new_max[:, None])

        acc = acc * alpha[:, None]
        denom = denom * alpha

        v_ptrs = (
            v_ptr
            + b * stride_v_b
            + h * stride_v_h
            + n_curr[:, None] * stride_v_n
            + offs_d[None, :] * stride_v_d
        )
        v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

        acc += tl.dot(probs, v)
        denom += tl.sum(probs, axis=1)
        max_logits = new_max
        n_start += BLOCK_N

    out = acc / denom[:, None]

    out_ptrs = (
        out_ptr
        + b * stride_o_b
        + h * stride_o_h
        + offs_m[:, None] * stride_o_m
        + offs_d[None, :] * stride_o_d
    )
    tl.store(out_ptrs, out, mask=m_mask[:, None])


@torch.fx.wrap
def fused_attention(q, k, v, scale: float):
    B = int(q.shape[0])
    H = int(q.shape[1])
    M = int(q.shape[2])
    D = int(q.shape[3])
    N = int(k.shape[3])

    out = torch.empty((B, H, M, D), device=q.device, dtype=q.dtype)

    if N <= 64:
        block_n = 32
    elif N <= 256:
        block_n = 64
    else:
        block_n = 128

    if M >= 4096:
        block_m = 16
    elif M >= 1024:
        block_m = 32
    else:
        block_m = 64 if D <= 32 else 32

    if D <= 32:
        num_warps = 4
    else:
        num_warps = 8 if block_m >= 32 else 4

    grid = (triton.cdiv(M, block_m), B * H)
    _fused_attention_kernel[grid](
        q, k, v, out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        H, M, N,
        scale,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        D_HEAD=D,
        num_warps=num_warps,
        num_stages=2,
    )
    return out