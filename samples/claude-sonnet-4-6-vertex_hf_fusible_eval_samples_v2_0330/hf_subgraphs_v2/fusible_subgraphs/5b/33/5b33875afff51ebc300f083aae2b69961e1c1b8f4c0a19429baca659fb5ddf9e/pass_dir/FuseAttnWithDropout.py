"""
Variant: softmax(dim=-1) + dropout(0.1,False,False) + matmul + permute(0,2,1,3)
Matches the case where the model FX graph DOES preserve the dropout node.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_smdp2_kernel(
    scores_ptr,
    values_ptr,
    output_ptr,
    B, H, S, D,
    s_stride_b, s_stride_h, s_stride_q, s_stride_k,
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    o_stride_b, o_stride_s, o_stride_h, o_stride_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // (H * S)
    tmp = pid % (H * S)
    h = tmp // S
    q = tmp % S

    scores_base = scores_ptr + b * s_stride_b + h * s_stride_h + q * s_stride_q
    k_range = tl.arange(0, BLOCK_S)
    k_mask = k_range < S

    scores = tl.load(
        scores_base + k_range * s_stride_k, mask=k_mask, other=0.0
    ).to(tl.float32)
    scores = tl.where(k_mask, scores, float('-inf'))

    scores_max = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - scores_max)
    attn = exp_s / tl.sum(exp_s, axis=0)

    values_base = values_ptr + b * v_stride_b + h * v_stride_h
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    v = tl.load(
        values_base + k_range[:, None] * v_stride_s + d_range[None, :] * v_stride_d,
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    acc = tl.sum(attn[:, None] * v, axis=0)

    out_base = output_ptr + b * o_stride_b + q * o_stride_s + h * o_stride_h
    tl.store(
        out_base + d_range * o_stride_d,
        acc.to(output_ptr.dtype.element_ty),
        mask=d_mask,
    )


@torch.fx.wrap
def fused_softmax_dropout_matmul_permute(tmp_1, in_3):
    B, H, S, _ = tmp_1.shape
    D = in_3.shape[-1]

    BLOCK_S = triton.next_power_of_2(S)
    BLOCK_D = triton.next_power_of_2(D)

    output = torch.empty(B, S, H, D, dtype=tmp_1.dtype, device=tmp_1.device)

    _fused_smdp2_kernel[(B * H * S,)](
        tmp_1, in_3, output,
        B, H, S, D,
        tmp_1.stride(0), tmp_1.stride(1), tmp_1.stride(2), tmp_1.stride(3),
        in_3.stride(0), in_3.stride(1), in_3.stride(2), in_3.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return output


def pattern(tmp_1, in_3):
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    return tmp_5


def replacement_args(tmp_1, in_3):
    return (tmp_1, in_3)


def replacement_func():
    return fused_softmax_dropout_matmul_permute