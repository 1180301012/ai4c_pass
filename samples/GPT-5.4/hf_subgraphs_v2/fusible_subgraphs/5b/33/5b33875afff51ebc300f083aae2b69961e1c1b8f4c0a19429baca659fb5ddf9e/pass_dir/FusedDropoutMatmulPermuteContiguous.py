import torch
import triton
import triton.language as tl


def pattern(probs, value):
    tmp_3 = torch.nn.functional.dropout(probs, 0.1, False, False)
    matmul = torch.matmul(tmp_3, value)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(probs, value):
    return (probs, value)


@triton.jit
def matmul_permute_kernel(
    probs_ptr,
    value_ptr,
    out_ptr,
    B,
    H,
    Q,
    K,
    D,
    probs_s0,
    probs_s1,
    probs_s2,
    probs_s3,
    value_s0,
    value_s1,
    value_s2,
    value_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    q = pid % Q
    t = pid // Q
    h = t % H
    b = t // H

    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    k_mask = offs_k < K
    d_mask = offs_d < D

    probs_ptrs = probs_ptr + b * probs_s0 + h * probs_s1 + q * probs_s2 + offs_k * probs_s3
    p = tl.load(probs_ptrs, mask=k_mask, other=0.0).to(tl.float32)

    value_ptrs = (
        value_ptr
        + b * value_s0
        + h * value_s1
        + offs_k[:, None] * value_s2
        + offs_d[None, :] * value_s3
    )
    v = tl.load(value_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(p[:, None] * v, axis=0)

    out_ptrs = out_ptr + b * out_s0 + q * out_s1 + h * out_s2 + offs_d * out_s3
    tl.store(out_ptrs, acc, mask=d_mask)


@torch.fx.wrap
def fused_dropout_matmul_permute_contiguous(probs, value):
    B = probs.shape[0]
    H = probs.shape[1]
    Q = probs.shape[2]
    K = probs.shape[3]
    D = value.shape[3]

    out = torch.empty((B, Q, H, D), device=value.device, dtype=value.dtype)

    if K <= 8:
        block_k = 8
    elif K <= 16:
        block_k = 16
    elif K <= 32:
        block_k = 32
    else:
        block_k = 64

    if D <= 8:
        block_d = 8
        num_warps = 1
    elif D <= 16:
        block_d = 16
        num_warps = 2
    elif D <= 32:
        block_d = 32
        num_warps = 4
    else:
        block_d = 64
        num_warps = 4

    grid = (B * H * Q,)
    matmul_permute_kernel[grid](
        probs,
        value,
        out,
        B,
        H,
        Q,
        K,
        D,
        probs.stride(0),
        probs.stride(1),
        probs.stride(2),
        probs.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_dropout_matmul_permute_contiguous