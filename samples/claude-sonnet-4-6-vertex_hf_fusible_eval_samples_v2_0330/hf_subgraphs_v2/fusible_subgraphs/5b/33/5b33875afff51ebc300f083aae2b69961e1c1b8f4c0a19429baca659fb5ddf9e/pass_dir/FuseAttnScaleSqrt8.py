"""
Fuse: in_0/2.8284271247461903 + mask_add + softmax + dropout(training=False) + matmul + permute(0,2,1,3) + contiguous
into a single Triton kernel that writes directly in the permuted layout.

Matches:
  tmp_0 = in_0 / 2.8284271247461903
  tmp_1 = tmp_0 + in_2
  tmp_2 = softmax(tmp_1, dim=-1)
  tmp_3 = dropout(tmp_2, 0.1, False, False)   # no-op at inference
  matmul = torch.matmul(tmp_3, in_3)
  tmp_5 = matmul.permute(0, 2, 1, 3)
  tmp_6 = tmp_5.contiguous()
  return tmp_6

Used by: tiny-random-ConvBertForSequenceClassification  (head_dim=8, scale=sqrt(8))
"""

import torch
import triton
import triton.language as tl

# 1 / 2.8284271247461903  (= 1/sqrt(8))
_INV_SCALE_SQRT8 = 0.35355339059327373


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_attn_sqrt8_kernel(
    scores_ptr,   # [B, H, S, S]
    mask_ptr,     # [1, 1, 1, S]  (broadcast mask)
    values_ptr,   # [B, H, S, D]
    output_ptr,   # [B, S, H, D]  (permuted layout, already contiguous)
    B, H, S, D,
    # strides for scores [B, H, S, S]
    s_stride_b, s_stride_h, s_stride_q, s_stride_k,
    # stride along last dim of mask (shape [1,1,1,S])
    m_stride_k,
    # strides for values [B, H, S, D]
    v_stride_b, v_stride_h, v_stride_s, v_stride_d,
    # strides for output [B, S, H, D]
    o_stride_b, o_stride_s, o_stride_h, o_stride_d,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """One program handles one (b, h, q) triplet → produces output[b, q, h, :]."""
    pid = tl.program_id(0)
    b = pid // (H * S)
    tmp = pid % (H * S)
    h = tmp // S
    q = tmp % S

    # ---- load & scale attention scores: scores[b, h, q, :] ----
    scores_base = scores_ptr + b * s_stride_b + h * s_stride_h + q * s_stride_q
    k_range = tl.arange(0, BLOCK_S)
    k_mask = k_range < S

    scores = tl.load(
        scores_base + k_range * s_stride_k, mask=k_mask, other=0.0
    ).to(tl.float32)
    # multiply by 1 / 2.8284271247461903 = 1/sqrt(8)
    scores = scores * 0.35355339059327373

    # ---- add attention mask: mask[0, 0, 0, :] ----
    attn_mask = tl.load(
        mask_ptr + k_range * m_stride_k, mask=k_mask, other=0.0
    ).to(tl.float32)
    scores = scores + attn_mask

    # zero-pad slots beyond S with -inf so softmax ignores them
    scores = tl.where(k_mask, scores, float('-inf'))

    # ---- softmax ----
    scores_max = tl.max(scores, axis=0)
    exp_s = tl.exp(scores - scores_max)
    attn = exp_s / tl.sum(exp_s, axis=0)   # [BLOCK_S]

    # ---- load values and compute weighted sum ----
    values_base = values_ptr + b * v_stride_b + h * v_stride_h
    d_range = tl.arange(0, BLOCK_D)
    d_mask = d_range < D

    # [BLOCK_S, BLOCK_D]
    v = tl.load(
        values_base + k_range[:, None] * v_stride_s + d_range[None, :] * v_stride_d,
        mask=k_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # weighted sum over S → [BLOCK_D]
    acc = tl.sum(attn[:, None] * v, axis=0)

    # ---- store to output[b, q, h, :] in permuted layout ----
    out_base = output_ptr + b * o_stride_b + q * o_stride_s + h * o_stride_h
    tl.store(
        out_base + d_range * o_stride_d,
        acc.to(output_ptr.dtype.element_ty),
        mask=d_mask,
    )


# ---------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_attn_scale_sqrt8(in_0, in_2, in_3):
    """
    in_0  : attention scores  [B, H, S, S]
    in_2  : attention mask    [1, 1, 1, S]
    in_3  : value tensor      [B, H, S, D]
    returns: context          [B, S, H, D]   (contiguous, permuted)
    """
    scores = in_0
    mask   = in_2
    values = in_3

    B, H, S, _ = scores.shape
    D = values.shape[-1]

    BLOCK_S = triton.next_power_of_2(S)
    BLOCK_D = triton.next_power_of_2(D)

    output = torch.empty(B, S, H, D, dtype=scores.dtype, device=scores.device)

    _fused_attn_sqrt8_kernel[(B * H * S,)](
        scores, mask, values, output,
        B, H, S, D,
        scores.stride(0), scores.stride(1), scores.stride(2), scores.stride(3),
        mask.stride(3),
        values.stride(0), values.stride(1), values.stride(2), values.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        BLOCK_S=BLOCK_S,
        BLOCK_D=BLOCK_D,
        num_warps=4,
    )

    return output


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------

def pattern(in_0, in_2, in_3):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_2, in_3):
    return (in_0, in_2, in_3)


def replacement_func():
    return fused_attn_scale_sqrt8