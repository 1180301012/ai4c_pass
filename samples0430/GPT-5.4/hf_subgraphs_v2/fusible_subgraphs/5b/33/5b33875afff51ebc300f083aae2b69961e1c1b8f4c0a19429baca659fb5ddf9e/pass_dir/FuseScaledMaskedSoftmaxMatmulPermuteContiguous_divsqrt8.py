import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0 / 2.8284271247461903
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_6 = matmul.permute(0, 2, 1, 3).contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_2, in_3)


@triton.jit
def _attention_kernel(
    scores_ptr,
    mask_ptr,
    value_ptr,
    out_ptr,
    B,
    H,
    S,
    D,
    SCALE,
    scores_s0,
    scores_s1,
    scores_s2,
    scores_s3,
    mask_s0,
    mask_s1,
    mask_s2,
    mask_s3,
    value_s0,
    value_s1,
    value_s2,
    value_s3,
    out_s0,
    out_s1,
    out_s2,
    out_s3,
    BLOCK_D: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    total = B * H * S
    if pid >= total:
        return

    s_idx = pid % S
    h_idx = (pid // S) % H
    b_idx = pid // (S * H)

    d_offsets = tl.arange(0, BLOCK_D)
    s_offsets = tl.arange(0, BLOCK_S)
    d_mask = d_offsets < D
    s_mask = s_offsets < S

    base_scores = b_idx * scores_s0 + h_idx * scores_s1 + s_idx * scores_s2
    logits = tl.load(scores_ptr + base_scores + s_offsets * scores_s3, mask=s_mask, other=-float("inf"))

    base_mask = b_idx * mask_s0
    mask_vals = tl.load(mask_ptr + base_mask + s_offsets * mask_s3, mask=s_mask, other=0.0)

    logits = logits / SCALE + mask_vals
    logits = tl.where(s_mask, logits, -float("inf"))
    m = tl.max(logits, axis=0)
    denom = tl.sum(tl.exp(logits - m), axis=0)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for start in range(0, 4096, BLOCK_S):
        offs = start + s_offsets
        cur_mask = offs < S
        if start >= S:
            break
        p = tl.load(scores_ptr + base_scores + offs * scores_s3, mask=cur_mask, other=-float("inf"))
        mv = tl.load(mask_ptr + base_mask + offs * mask_s3, mask=cur_mask, other=0.0)
        p = tl.exp((p / SCALE + mv) - m) / denom

        val_ptrs = (
            value_ptr
            + b_idx * value_s0
            + h_idx * value_s1
            + offs[:, None] * value_s2
            + d_offsets[None, :] * value_s3
        )
        vals = tl.load(val_ptrs, mask=cur_mask[:, None] & d_mask[None, :], other=0.0)
        acc += tl.sum(vals.to(tl.float32) * p[:, None].to(tl.float32), axis=0)

    out_ptrs = out_ptr + b_idx * out_s0 + s_idx * out_s1 + h_idx * out_s2 + d_offsets * out_s3
    tl.store(out_ptrs, acc, mask=d_mask)


@torch.fx.wrap
def fused_attention_div8(scores, mask, value):
    B = scores.shape[0]
    H = scores.shape[1]
    S = scores.shape[2]
    D = value.shape[3]
    out = torch.empty((B, S, H, D), device=scores.device, dtype=scores.dtype)
    grid = (B * H * S,)
    block_d = 8
    while block_d < D:
        block_d *= 2
    if block_d < 16:
        block_d = 16
    block_s = 16
    if S > 16:
        block_s = 32
    if S > 32:
        block_s = 64
    _attention_kernel[grid](
        scores,
        mask,
        value,
        out,
        B,
        H,
        S,
        D,
        2.8284271247461903,
        scores.stride(0),
        scores.stride(1),
        scores.stride(2),
        scores.stride(3),
        mask.stride(0),
        mask.stride(1),
        mask.stride(2),
        mask.stride(3),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        value.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_D=block_d,
        BLOCK_S=block_s,
    )
    return out


def replacement_func():
    return fused_attention_div8