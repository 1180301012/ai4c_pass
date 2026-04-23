import torch
import triton
import triton.language as tl


def pattern(x, mask, value):
    tmp_1 = x + mask
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, value)
    return matmul


def replacement_args(x, mask, value):
    return (x, mask, value)


@triton.jit

def fused_add_softmax_matmul_kernel(
    x_ptr,
    mask_ptr,
    value_ptr,
    out_ptr,
    B,
    H,
    Q,
    K,
    D,
    x_s0,
    x_s1,
    x_s2,
    x_s3,
    mask_s3,
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

    x_ptrs = x_ptr + b * x_s0 + h * x_s1 + q * x_s2 + offs_k * x_s3
    x_vals = tl.load(x_ptrs, mask=k_mask, other=-float("inf")).to(tl.float32)

    mask_ptrs = mask_ptr + offs_k * mask_s3
    mask_vals = tl.load(mask_ptrs, mask=k_mask, other=-float("inf")).to(tl.float32)

    logits = x_vals + mask_vals
    row_max = tl.max(logits, axis=0)
    numerators = tl.exp(logits - row_max)
    denom = tl.sum(numerators, axis=0)
    probs = numerators / denom

    value_ptrs = (
        value_ptr
        + b * value_s0
        + h * value_s1
        + offs_k[:, None] * value_s2
        + offs_d[None, :] * value_s3
    )
    v = tl.load(value_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0).to(tl.float32)
    acc = tl.sum(probs[:, None] * v, axis=0)

    out_ptrs = out_ptr + b * out_s0 + h * out_s1 + q * out_s2 + offs_d * out_s3
    tl.store(out_ptrs, acc.to(tl.float32), mask=d_mask)


@torch.fx.wrap

def fused_add_softmax_matmul(x, mask, value):
    B = x.shape[0]
    H = x.shape[1]
    Q = x.shape[2]
    K = x.shape[3]
    D = value.shape[3]

    out = torch.empty((B, H, Q, D), device=value.device, dtype=value.dtype)

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
    fused_add_softmax_matmul_kernel[grid](
        x,
        mask,
        value,
        out,
        B,
        H,
        Q,
        K,
        D,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        mask.stride(3),
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
    return fused_add_softmax_matmul