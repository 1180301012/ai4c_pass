import torch
import triton
import triton.language as tl


@triton.jit
def fused_rope_kernel(
    pos_ptr,
    cos_ptr,
    cls_q_ptr,
    q_ptr,
    k_ptr,
    sin_ptr,
    out_k_ptr,
    out_q_ptr,
    H,
    T,
    pos_stride_s,
    pos_stride_d,
    cos_stride_s,
    cos_stride_d,
    cls_q_stride_h,
    cls_q_stride_d,
    q_stride_h,
    q_stride_t,
    q_stride_d,
    k_stride_h,
    k_stride_t,
    k_stride_d,
    sin_stride_s,
    sin_stride_d,
    out_k_stride_h,
    out_k_stride_t,
    out_k_stride_d,
    out_q_stride_h,
    out_q_stride_t,
    out_q_stride_d,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid = tl.program_id(0)
    row_ids = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    d = tl.arange(0, BLOCK_D)
    total_rows = H * T

    valid_rows = row_ids < total_rows
    heads = row_ids // T
    tokens = row_ids % T
    body_rows = tokens > 0
    body_tokens = tokens - 1
    mate_d = d ^ 1

    mask = valid_rows[:, None]
    body_mask = valid_rows[:, None] & body_rows[:, None]

    cls_q_ptrs = cls_q_ptr + heads[:, None] * cls_q_stride_h + d[None, :] * cls_q_stride_d
    cls_q = tl.load(cls_q_ptrs, mask=mask, other=0.0)

    cls_k_ptrs = k_ptr + heads[:, None] * k_stride_h + d[None, :] * k_stride_d
    cls_k = tl.load(cls_k_ptrs, mask=mask, other=0.0)

    q_body_base = q_ptr + heads[:, None] * q_stride_h + body_tokens[:, None] * q_stride_t
    q_val = tl.load(q_body_base + d[None, :] * q_stride_d, mask=body_mask, other=0.0)
    q_mate = tl.load(q_body_base + mate_d[None, :] * q_stride_d, mask=body_mask, other=0.0)
    cos_val = tl.load(cos_ptr + body_tokens[:, None] * cos_stride_s + d[None, :] * cos_stride_d, mask=body_mask, other=0.0)
    sin_val = tl.load(sin_ptr + body_tokens[:, None] * sin_stride_s + d[None, :] * sin_stride_d, mask=body_mask, other=0.0)
    q_rot = tl.where((d[None, :] & 1) == 0, -q_mate, q_mate)
    q_body = q_val * cos_val + q_rot * sin_val
    q_out = tl.where((tokens == 0)[:, None], cls_q, q_body)

    out_q_ptrs = out_q_ptr + heads[:, None] * out_q_stride_h + tokens[:, None] * out_q_stride_t + d[None, :] * out_q_stride_d
    tl.store(out_q_ptrs, q_out, mask=mask)

    k_body_base = k_ptr + heads[:, None] * k_stride_h + tokens[:, None] * k_stride_t
    k_val = tl.load(k_body_base + d[None, :] * k_stride_d, mask=body_mask, other=0.0)
    k_mate = tl.load(k_body_base + mate_d[None, :] * k_stride_d, mask=body_mask, other=0.0)
    pos0 = tl.load(pos_ptr + body_tokens[:, None] * pos_stride_s + d[None, :] * pos_stride_d, mask=body_mask, other=0.0)
    pos1 = tl.load(pos_ptr + body_tokens[:, None] * pos_stride_s + (BLOCK_D + d)[None, :] * pos_stride_d, mask=body_mask, other=0.0)
    k_rot = tl.where((d[None, :] & 1) == 0, -k_mate, k_mate)
    k_body = k_val * pos1 + k_rot * pos0
    k_out = tl.where((tokens == 0)[:, None], cls_k, k_body)

    out_k_ptrs = out_k_ptr + heads[:, None] * out_k_stride_h + tokens[:, None] * out_k_stride_t + d[None, :] * out_k_stride_d
    tl.store(out_k_ptrs, k_out, mask=mask)


@torch.fx.wrap
def fused_rope(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    h = int(in_3.shape[1])
    s = int(in_3.shape[2])
    t = s + 1

    out_k = torch.empty_like(in_6)
    out_q = torch.empty_like(in_6)

    grid = (triton.cdiv(h * t, 4),)
    fused_rope_kernel[grid](
        in_0,
        in_1,
        in_2,
        in_3,
        in_4,
        in_5,
        out_k,
        out_q,
        h,
        t,
        int(in_0.stride(0)),
        int(in_0.stride(1)),
        int(in_1.stride(0)),
        int(in_1.stride(1)),
        int(in_2.stride(1)),
        int(in_2.stride(3)),
        int(in_3.stride(1)),
        int(in_3.stride(2)),
        int(in_3.stride(3)),
        int(in_4.stride(1)),
        int(in_4.stride(2)),
        int(in_4.stride(3)),
        int(in_5.stride(0)),
        int(in_5.stride(1)),
        int(out_k.stride(1)),
        int(out_k.stride(2)),
        int(out_k.stride(3)),
        int(out_q.stride(1)),
        int(out_q.stride(2)),
        int(out_q.stride(3)),
        BLOCK_ROWS=4,
        BLOCK_D=64,
        num_warps=4,
        num_stages=2,
    )
    return out_k, out_q


def replacement_func():
    return fused_rope