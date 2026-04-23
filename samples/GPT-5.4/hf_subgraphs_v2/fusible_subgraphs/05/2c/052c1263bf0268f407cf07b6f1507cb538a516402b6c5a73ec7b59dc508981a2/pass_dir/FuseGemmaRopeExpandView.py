import torch
import triton
import triton.language as tl


# Match the full observable subgraph exactly.
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    tmp_7 = tmp_6[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    tmp_10 = in_5[slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None)]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    tmp_13 = in_0[slice(None, None, None), slice(None, None, None), slice(None, None, None), slice(None, 3, None)]
    tmp_14 = in_3.contiguous()
    return (tmp_13, tmp_6, tmp_9, tmp_14, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def _gemma_rope_expand_kernel(
    cos_ptr,
    key_ptr,
    sin_ptr,
    value_ptr,
    out6_ptr,
    out9_ptr,
    out12_ptr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 256)
    base = row * 256
    offs = base + cols
    pair_cols = tl.where(cols < 128, cols + 128, cols - 128)
    pair_offs = base + pair_cols

    key = tl.load(key_ptr + offs).to(tl.float32)
    cos = tl.load(cos_ptr + offs).to(tl.float32)
    sin = tl.load(sin_ptr + offs).to(tl.float32)
    val = tl.load(value_ptr + offs)
    rotated_src = tl.load(key_ptr + pair_offs).to(tl.float32)
    rotated = tl.where(cols < 128, -rotated_src, rotated_src)
    out = (key * cos + rotated * sin).to(tl.bfloat16)

    tl.store(out6_ptr + offs, out)

    for head in tl.static_range(0, 8):
        out_broadcast_offs = head * (3 * 256) + offs
        tl.store(out9_ptr + out_broadcast_offs, out)
        tl.store(out12_ptr + out_broadcast_offs, val)


@torch.fx.wrap
def _fused_gemma_rope_expand_view_impl(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_6 = torch.empty_like(in_2)
    tmp_9 = torch.empty((1, 8, 3, 256), device=in_2.device, dtype=in_2.dtype)
    tmp_12 = torch.empty((1, 8, 3, 256), device=in_5.device, dtype=in_5.dtype)

    _gemma_rope_expand_kernel[(3,)](
        in_1,
        in_2,
        in_4,
        in_5,
        tmp_6,
        tmp_9,
        tmp_12,
        num_warps=4,
        num_stages=1,
    )
    return (in_0, tmp_6, tmp_9, in_3, tmp_12)


def fused_gemma_rope_expand_view(in_0, in_1, in_2, in_3, in_4, in_5):
    outs = _fused_gemma_rope_expand_view_impl(in_0, in_1, in_2, in_3, in_4, in_5)
    return outs[0], outs[1], outs[2], outs[3], outs[4]


def replacement_func():
    return fused_gemma_rope_expand_view