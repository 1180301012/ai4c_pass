import torch
import triton
import triton.language as tl


# ----------- Kernels for standard attention shapes with O = 32 or 64 -----------
# We fuse:
#   scores = q @ k
#   probs  = softmax(scores * scale)
#   out    = probs @ v
# and directly write the final layout equivalent to:
#   out.permute(0, 2, 1, 3).contiguous().view(B, M, H * D)
# where input q is [B, H, M, D], k is [B, H, D, N], v is [B, H, N, D].


@triton.jit
def _attn_fwd_o32_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d, stride_k_n,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_m, stride_o_hd,
    B, H, M, N,
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

    m_i = offs_m[:, None]
    max_logits = tl.full((BLOCK_M,), -float('inf'), tl.float32)
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

        logits = tl.dot(q, k)
        logits = logits * scale
        logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, -float('inf'))

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_logits, block_max)
        alpha = tl.exp(max_logits - new_max)
        p = tl.exp(logits - new_max[:, None])

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

        acc += tl.dot(p, v)
        denom += tl.sum(p, axis=1)
        max_logits = new_max
        n_start += BLOCK_N

    out = acc / denom[:, None]

    hd_base = h * D_HEAD + offs_d
    out_ptrs = (
        out_ptr
        + b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + hd_base[None, :] * stride_o_hd
    )
    tl.store(out_ptrs, out, mask=m_mask[:, None])


@triton.jit
def _attn_fwd_o64_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d, stride_k_n,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_m, stride_o_hd,
    B, H, M, N,
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

    max_logits = tl.full((BLOCK_M,), -float('inf'), tl.float32)
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

        logits = tl.dot(q, k)
        logits = logits * scale
        logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, -float('inf'))

        block_max = tl.max(logits, axis=1)
        new_max = tl.maximum(max_logits, block_max)
        alpha = tl.exp(max_logits - new_max)
        p = tl.exp(logits - new_max[:, None])

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

        acc += tl.dot(p, v)
        denom += tl.sum(p, axis=1)
        max_logits = new_max
        n_start += BLOCK_N

    out = acc / denom[:, None]

    hd_base = h * D_HEAD + offs_d
    out_ptrs = (
        out_ptr
        + b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + hd_base[None, :] * stride_o_hd
    )
    tl.store(out_ptrs, out, mask=m_mask[:, None])


# ----------- Tiny special-case kernels for very small attention graphs -----------
# Shapes targeted:
#   q [B,H,M,D], k [B,H,D,N], v [B,H,N,D]
# with M,N <= 16 and D <= 48. These cases are tiny enough that a single program
# can handle one (b,h) pair.


@triton.jit
def _attn_tiny_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_q_b, stride_q_h, stride_q_m, stride_q_d,
    stride_k_b, stride_k_h, stride_k_d, stride_k_n,
    stride_v_b, stride_v_h, stride_v_n, stride_v_d,
    stride_o_b, stride_o_m, stride_o_hd,
    H, M, N,
    scale,
    MAX_M: tl.constexpr,
    MAX_N: tl.constexpr,
    D_HEAD: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // H
    h = pid % H

    offs_m = tl.arange(0, MAX_M)
    offs_n = tl.arange(0, MAX_N)
    offs_d = tl.arange(0, D_HEAD)

    m_mask = offs_m < M
    n_mask = offs_n < N

    q_ptrs = (
        q_ptr
        + b * stride_q_b
        + h * stride_q_h
        + offs_m[:, None] * stride_q_m
        + offs_d[None, :] * stride_q_d
    )
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    k_ptrs = (
        k_ptr
        + b * stride_k_b
        + h * stride_k_h
        + offs_d[:, None] * stride_k_d
        + offs_n[None, :] * stride_k_n
    )
    k = tl.load(k_ptrs, mask=n_mask[None, :], other=0.0).to(tl.float32)

    logits = tl.dot(q, k) * scale
    logits = tl.where(m_mask[:, None] & n_mask[None, :], logits, -float('inf'))

    row_max = tl.max(logits, axis=1)
    p = tl.exp(logits - row_max[:, None])
    denom = tl.sum(p, axis=1)
    p = p / denom[:, None]

    v_ptrs = (
        v_ptr
        + b * stride_v_b
        + h * stride_v_h
        + offs_n[:, None] * stride_v_n
        + offs_d[None, :] * stride_v_d
    )
    v = tl.load(v_ptrs, mask=n_mask[:, None], other=0.0).to(tl.float32)

    out = tl.dot(p, v)

    hd_base = h * D_HEAD + offs_d
    out_ptrs = (
        out_ptr
        + b * stride_o_b
        + offs_m[:, None] * stride_o_m
        + hd_base[None, :] * stride_o_hd
    )
    tl.store(out_ptrs, out, mask=m_mask[:, None])


# ----------- Python helpers -----------


def _canonical_dims(q, k, v):
    B = int(q.shape[0])
    H = int(q.shape[1])
    M = int(q.shape[2])
    D = int(q.shape[3])
    N = int(k.shape[3])
    assert k.shape[0] == B and k.shape[1] == H and k.shape[2] == D
    assert v.shape[0] == B and v.shape[1] == H and v.shape[2] == N and v.shape[3] == D
    return B, H, M, N, D


def _output_tensor(q):
    B, H, M, _, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3], q.shape[3]
    return torch.empty((B, M, H * D), device=q.device, dtype=q.dtype)


@torch.fx.wrap
def fused_attention_dispatch(q, k, v, scale: float, route: str):
    B, H, M, N, D = _canonical_dims(q, k, v)
    out = torch.empty((B, M, H * D), device=q.device, dtype=q.dtype)

    # Tiny cases: avoid heavy tiling overhead.
    if M <= 16 and N <= 16 and D <= 48:
        max_m = 16
        max_n = 16
        grid = (B * H,)
        _attn_tiny_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            H, M, N,
            scale,
            MAX_M=max_m,
            MAX_N=max_n,
            D_HEAD=D,
        )
        return out

    num_warps = 4
    if D == 32:
        block_m = 32 if M >= 512 else 16
        block_n = 64 if N >= 128 else 32
        grid = (triton.cdiv(M, block_m), B * H)
        _attn_fwd_o32_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            B, H, M, N,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            D_HEAD=32,
            num_warps=num_warps,
            num_stages=2,
        )
        return out

    if D == 64:
        block_m = 16 if M >= 512 else 8
        block_n = 64 if N >= 128 else 32
        grid = (triton.cdiv(M, block_m), B * H)
        _attn_fwd_o64_kernel[grid](
            q, k, v, out,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            out.stride(0), out.stride(1), out.stride(2),
            B, H, M, N,
            scale,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            D_HEAD=64,
            num_warps=4 if block_m <= 16 else 8,
            num_stages=2,
        )
        return out

    # Conservative fallback route for unsupported head sizes.
    # We intentionally keep this path only for uncommon leftover shapes.
    scores = q @ k
    scores = scores / (1.0 / scale)
    probs = torch.softmax(scores, dim=-1)
    out_ref = probs @ v
    out_ref = out_ref.permute(0, 2, 1, 3).contiguous().view(B, M, H * D)
    out.copy_(out_ref)
    return out