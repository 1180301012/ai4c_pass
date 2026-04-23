import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_2 = torch.nn.functional.embedding(in_0, in_1, 0, None, 2.0, False, False)
    tmp_3 = tmp_2[(slice(None, None, None), slice(1, None, None))]
    tmp_4 = torch._C._nn.pad(tmp_3, [0, 0, 0, 1, 0, 0], 'constant', 0.0)
    tmp_5 = tmp_2[(slice(None, None, None), slice(None, -1, None))]
    tmp_6 = torch._C._nn.pad(tmp_5, [0, 0, 1, 0, 0, 0], 'constant', 0.0)
    tmp_7 = torch.cat([tmp_4, tmp_2, tmp_6], dim = 2)
    return tmp_7


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_H': 64}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_H': 128}, num_warps=4, num_stages=2),
    ],
    key=['H'],
)
@triton.jit

def _embedding_shift_cat_kernel(
    ids_ptr,
    weight_ptr,
    out_ptr,
    B,
    S,
    H,
    ids_stride_0,
    ids_stride_1,
    weight_stride_0,
    weight_stride_1,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    BLOCK_H: tl.constexpr,
):
    pid_bs = tl.program_id(0)
    pid_h = tl.program_id(1)

    b = pid_bs // S
    s = pid_bs % S
    h_offs = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    h_mask = h_offs < H

    ids_base = ids_ptr + b * ids_stride_0
    cur_id = tl.load(ids_base + s * ids_stride_1)

    next_valid = s + 1 < S
    prev_valid = s > 0
    next_id = tl.load(ids_base + (s + 1) * ids_stride_1, mask=next_valid, other=0)
    prev_id = tl.load(ids_base + (s - 1) * ids_stride_1, mask=prev_valid, other=0)

    cur_row = weight_ptr + tl.cast(cur_id, tl.int64) * weight_stride_0 + h_offs * weight_stride_1
    next_row = weight_ptr + tl.cast(next_id, tl.int64) * weight_stride_0 + h_offs * weight_stride_1
    prev_row = weight_ptr + tl.cast(prev_id, tl.int64) * weight_stride_0 + h_offs * weight_stride_1

    cur_val = tl.load(cur_row, mask=h_mask, other=0.0)
    next_val = tl.load(next_row, mask=h_mask & next_valid, other=0.0)
    prev_val = tl.load(prev_row, mask=h_mask & prev_valid, other=0.0)

    out_base = out_ptr + b * out_stride_0 + s * out_stride_1
    tl.store(out_base + h_offs * out_stride_2, next_val, mask=h_mask)
    tl.store(out_base + (H + h_offs) * out_stride_2, cur_val, mask=h_mask)
    tl.store(out_base + ((2 * H) + h_offs) * out_stride_2, prev_val, mask=h_mask)


@torch.fx.wrap
def fused_embedding_shift_cat(ids, weight):
    b = ids.shape[0]
    s = ids.shape[1]
    h = weight.shape[1]
    out = torch.empty((b, s, 3 * h), device=weight.device, dtype=weight.dtype)

    grid = lambda META: (b * s, triton.cdiv(h, META['BLOCK_H']))
    _embedding_shift_cat_kernel[grid](
        ids,
        weight,
        out,
        b,
        s,
        h,
        ids.stride(0),
        ids.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        out.stride(2),
    )
    return out


def replacement_func():
    return fused_embedding_shift_cat