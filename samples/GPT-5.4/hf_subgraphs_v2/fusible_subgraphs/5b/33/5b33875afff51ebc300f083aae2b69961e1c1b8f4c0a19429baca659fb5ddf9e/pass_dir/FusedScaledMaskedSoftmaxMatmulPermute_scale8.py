import torch
import triton
import triton.language as tl


def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3, "scale8")


@triton.jit
def fused_attention_permute_kernel(
    scores_ptr,
    mask_ptr,
    value_ptr,
    out_ptr,
    B,
    H,
    Q,
    K,
    D,
    scores_s0,
    scores_s1,
    scores_s2,
    scores_s3,
    mask_s3,
    value_s0,
    value_s1,
    value_s2,
    value_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    scale,
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

    score_ptrs = scores_ptr + b * scores_s0 + h * scores_s1 + q * scores_s2 + offs_k * scores_s3
    scores = tl.load(score_ptrs, mask=k_mask, other=-float("inf")).to(tl.float32)

    mask_ptrs = mask_ptr + offs_k * mask_s3
    additive_mask = tl.load(mask_ptrs, mask=k_mask, other=-float("inf")).to(tl.float32)

    logits = scores / scale + additive_mask
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
    v = tl.load(value_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)
    acc = tl.sum(probs[:, None] * v, axis=0)

    out_ptrs = out_ptr + b * out_s0 + q * out_s1 + h * out_s2 + offs_d * out_s3
    tl.store(out_ptrs, acc, mask=d_mask)


@torch.fx.wrap

def fused_scaled_masked_softmax_matmul_permute(scores, mask, value, route):
    B = scores.shape[0]
    H = scores.shape[1]
    Q = scores.shape[2]
    K = scores.shape[3]
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

    if route == "scale8":
        scale = 8.0
    else:
        scale = 2.8284271247461903

    grid = (B * H * Q,)
    fused_attention_permute_kernel[grid](
        scores,
        mask,
        value,
        out,
        B,
        H,
        Q,
        K,
        D,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        mask.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        scale,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        num_warps=num_warps,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_scaled_masked_softmax_matmul_permute