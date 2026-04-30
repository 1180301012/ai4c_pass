import torch
import triton
import triton.language as tl


def pattern(probs, value):
    matmul = torch.matmul(probs, value)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(probs, value):
    return (probs, value)


@triton.jit
def _attn_matmul_transpose_kernel(
    probs_ptr,
    value_ptr,
    out_ptr,
    H,
    S,
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
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    h_idx = pid % H
    b_idx = pid // H

    s_offsets = tl.arange(0, BLOCK_S)
    k_offsets = tl.arange(0, BLOCK_S)
    d_offsets = tl.arange(0, BLOCK_D)

    s_mask = s_offsets < S
    k_mask = k_offsets < S
    d_mask = d_offsets < D

    prob_ptrs = (
        probs_ptr
        + b_idx * probs_s0
        + h_idx * probs_s1
        + s_offsets[:, None] * probs_s2
        + k_offsets[None, :] * probs_s3
    )
    probs = tl.load(prob_ptrs, mask=s_mask[:, None] & k_mask[None, :], other=0.0)

    value_ptrs = (
        value_ptr
        + b_idx * value_s0
        + h_idx * value_s1
        + k_offsets[:, None] * value_s2
        + d_offsets[None, :] * value_s3
    )
    value = tl.load(value_ptrs, mask=k_mask[:, None] & d_mask[None, :], other=0.0)

    out = tl.dot(probs, value)

    out_ptrs = (
        out_ptr
        + b_idx * out_s0
        + s_offsets[:, None] * out_s1
        + h_idx * out_s2
        + d_offsets[None, :] * out_s3
    )
    tl.store(out_ptrs, out, mask=s_mask[:, None] & d_mask[None, :])


@torch.fx.wrap
def fused_attn_matmul_transpose(probs, value):
    B = probs.shape[0]
    H = probs.shape[1]
    S = probs.shape[2]
    D = value.shape[3]
    out = torch.empty((B, S, H, D), device=probs.device, dtype=probs.dtype)

    block_s = 16
    if S > 16:
        block_s = 32
    if S > 32:
        block_s = 64

    block_d = 8
    while block_d < D:
        block_d *= 2
    if block_d < 16:
        block_d = 16
    if block_d > 128:
        block_d = 128

    grid = (B * H,)
    _attn_matmul_transpose_kernel[grid](
        probs,
        value,
        out,
        H,
        S,
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
        BLOCK_S=block_s,
        BLOCK_D=block_d,
    )
    return out


def replacement_func():
    return fused_attn_matmul_transpose