import torch
import triton
import triton.language as tl


def pattern(scores, mask, value, scale):
    tmp_0 = scores / scale
    tmp_1 = tmp_0 + mask
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    matmul = torch.matmul(tmp_2, value)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(scores, mask, value, scale):
    return (scores, mask, value, scale)


@triton.jit
def fused_attention_kernel(
    scores_ptr, mask_ptr, value_ptr, output_ptr,
    scale,
    B, H, S, D,
    stride_s_b, stride_s_h, stride_s_q, stride_s_k,
    stride_m_k,
    stride_v_b, stride_v_h, stride_v_s, stride_v_d,
    stride_o_b, stride_o_q, stride_o_h, stride_o_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Fused kernel for: scale + mask + softmax + matmul + permute + contiguous
    Each program handles one (batch, head, query_seq) triple.
    Output is written directly in permuted layout [B, S, H, D].
    """
    pid = tl.program_id(0)
    q = pid % S
    h = (pid // S) % H
    b = pid // (S * H)

    k_offsets = tl.arange(0, BLOCK_S)
    k_mask = k_offsets < S

    # Load scores row: scores[b, h, q, k]
    scores_row = tl.load(
        scores_ptr + b * stride_s_b + h * stride_s_h + q * stride_s_q + k_offsets * stride_s_k,
        mask=k_mask, other=0.0,
    ).to(tl.float32)

    # Load mask row: mask is broadcast from [1,1,1,S], so only use last dim stride
    mask_row = tl.load(
        mask_ptr + k_offsets * stride_m_k,
        mask=k_mask, other=0.0,
    ).to(tl.float32)

    # Scale and add mask
    scaled = scores_row / scale + mask_row

    # Mask out invalid positions for softmax (set to -inf so exp gives ~0)
    scaled = tl.where(k_mask, scaled, float("-inf"))

    # Softmax: subtract max for numerical stability
    max_val = tl.max(scaled, axis=0)
    exp_vals = tl.exp(scaled - max_val)
    sum_exp = tl.sum(exp_vals, axis=0)
    probs = exp_vals / sum_exp
    # probs at invalid positions are 0 (since exp(-inf) ≈ 0)

    # Matmul: output[b, q, h, d] = sum_k probs[k] * value[b, h, k, d]
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < D

    # Load value block: value[b, h, k_offsets, d_offsets] -> [BLOCK_S, BLOCK_D]
    v_base = value_ptr + b * stride_v_b + h * stride_v_h
    v_ptrs = v_base + k_offsets[:, None] * stride_v_s + d_offsets[None, :] * stride_v_d

    value_block = tl.load(
        v_ptrs,
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Compute weighted sum: sum_k probs[k] * value[k, d]
    acc = tl.sum(probs[:, None] * value_block, axis=0)

    # Store output in permuted layout [B, S, H, D]
    o_ptrs = output_ptr + b * stride_o_b + q * stride_o_q + h * stride_o_h + d_offsets * stride_o_d
    tl.store(o_ptrs, acc, mask=d_mask)


def _next_power_of_2(n):
    if n <= 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p


@torch.fx.wrap
def fused_attention(scores, mask, value, scale):
    B, H, S, K = scores.shape
    _, _, _, D = value.shape

    scale_val = float(scale)

    BLOCK_S = _next_power_of_2(S)
    BLOCK_D = _next_power_of_2(D)

    # Create output in permuted layout [B, S, H, D]
    output = torch.empty((B, S, H, D), dtype=scores.dtype, device=scores.device)

    # Grid: each program handles one (b, h, q) triple
    grid = (B * H * S,)

    fused_attention_kernel[grid](
        scores_ptr=scores,
        mask_ptr=mask,
        value_ptr=value,
        output_ptr=output,
        scale=scale_val,
        B=B,
        H=H,
        S=S,
        D=D,
        stride_s_b=scores.stride()[0],
        stride_s_h=scores.stride()[1],
        stride_s_q=scores.stride()[2],
        stride_s_k=scores.stride()[3],
        stride_m_k=mask.stride()[-1],
        stride_v_b=value.stride()[0],
        stride_v_h=value.stride()[1],
        stride_v_s=value.stride()[2],
        stride_v_d=value.stride()[3],
        stride_o_b=output.stride()[0],
        stride_o_q=output.stride()[1],
        stride_o_h=output.stride()[2],
        stride_o_d=output.stride()[3],
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
    )

    return output


def replacement_func():
    return fused_attention