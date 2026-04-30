import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_0 = in_2 * in_1
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    tmp_7 = tmp_6[
        slice(None, None, None),
        slice(None, None, None),
        None,
        slice(None, None, None),
        slice(None, None, None),
    ]
    tmp_8 = tmp_7.expand(1, 1, 8, 3, 256)
    tmp_9 = tmp_8.reshape(1, 8, 3, 256)
    tmp_10 = in_5[
        slice(None, None, None),
        slice(None, None, None),
        None,
        slice(None, None, None),
        slice(None, None, None),
    ]
    tmp_11 = tmp_10.expand(1, 1, 8, 3, 256)
    tmp_12 = tmp_11.reshape(1, 8, 3, 256)
    tmp_13 = in_0[
        slice(None, None, None),
        slice(None, None, None),
        slice(None, None, None),
        slice(None, 3, None),
    ]
    tmp_14 = in_3.contiguous()
    return (tmp_13, tmp_6, tmp_9, tmp_14, tmp_12)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.jit
def _fused_rotary_key_kernel(
    key_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    key_stride_s,
    key_stride_d,
    cos_stride_s,
    cos_stride_d,
    sin_stride_s,
    sin_stride_d,
    out_stride_s,
    out_stride_d,
    BLOCK_D: tl.constexpr,
    HALF_D: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)

    key_ptrs = key_ptr + pid * key_stride_s + offs_d * key_stride_d
    cos_ptrs = cos_ptr + pid * cos_stride_s + offs_d * cos_stride_d
    sin_ptrs = sin_ptr + pid * sin_stride_s + offs_d * sin_stride_d

    rot_offs_d = tl.where(offs_d < HALF_D, offs_d + HALF_D, offs_d - HALF_D)
    rot_sign = tl.where(offs_d < HALF_D, -1.0, 1.0)
    rot_ptrs = key_ptr + pid * key_stride_s + rot_offs_d * key_stride_d

    key = tl.load(key_ptrs).to(tl.float32)
    cos = tl.load(cos_ptrs).to(tl.float32)
    sin = tl.load(sin_ptrs).to(tl.float32)
    rot = tl.load(rot_ptrs).to(tl.float32)

    out = key * cos + rot * sin * rot_sign

    out_ptrs = out_ptr + pid * out_stride_s + offs_d * out_stride_d
    tl.store(out_ptrs, out)


@torch.fx.wrap
def fused_gemma_rotary_key_and_views_impl(in_0, in_1, in_2, in_3, in_4, in_5):
    out_6 = torch.empty_like(in_2)

    n_seq = in_2.shape[2]
    _fused_rotary_key_kernel[(n_seq,)](
        in_2,
        in_1,
        in_4,
        out_6,
        in_2.stride(2),
        in_2.stride(3),
        in_1.stride(2),
        in_1.stride(3),
        in_4.stride(2),
        in_4.stride(3),
        out_6.stride(2),
        out_6.stride(3),
        BLOCK_D=256,
        HALF_D=128,
        num_warps=4,
        num_stages=1,
    )

    tmp_9 = out_6[:, :, None, :, :].expand(1, 1, 8, 3, 256).reshape(1, 8, 3, 256)
    tmp_12 = in_5[:, :, None, :, :].expand(1, 1, 8, 3, 256).reshape(1, 8, 3, 256)
    tmp_13 = in_0[:, :, :, :3]
    tmp_14 = in_3.contiguous()
    return (tmp_13, out_6, tmp_9, tmp_14, tmp_12)


def fused_gemma_rotary_key_and_views(in_0, in_1, in_2, in_3, in_4, in_5):
    outs = fused_gemma_rotary_key_and_views_impl(in_0, in_1, in_2, in_3, in_4, in_5)
    return outs[0], outs[1], outs[2], outs[3], outs[4]


def replacement_func():
    return fused_gemma_rotary_key_and_views