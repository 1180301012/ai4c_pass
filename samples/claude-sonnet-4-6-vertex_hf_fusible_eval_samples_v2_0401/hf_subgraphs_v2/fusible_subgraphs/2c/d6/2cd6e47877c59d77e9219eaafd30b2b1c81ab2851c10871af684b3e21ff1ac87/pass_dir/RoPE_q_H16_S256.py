import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_q16s256(
    x_ptr, cos_ptr, sin_ptr, cls_ptr, out_ptr,
    H, S,
    stride_xh, stride_xs,
    stride_cs,
    stride_outh, stride_outs,
    stride_clsh,
    D: tl.constexpr,
):
    h = tl.program_id(0)
    s_out = tl.program_id(1)
    d = tl.arange(0, D)
    out_base = h * stride_outh + s_out * stride_outs

    if s_out == 0:
        cls = tl.load(cls_ptr + h * stride_clsh + d)
        tl.store(out_ptr + out_base + d, cls)
    else:
        s = s_out - 1
        x_base = h * stride_xh + s * stride_xs
        x = tl.load(x_ptr + x_base + d)
        c = tl.load(cos_ptr + s * stride_cs + d)
        sv = tl.load(sin_ptr + s * stride_cs + d)
        x_p = tl.load(x_ptr + x_base + (d ^ 1))
        is_even = (d % 2) == 0
        rotate = tl.where(is_even, -x_p, x_p)
        tl.store(out_ptr + out_base + d, x * c + rotate * sv)


@torch.fx.wrap
def rope_q_fwd_16_256(in_3, in_1, in_5, in_2, in_6):
    B, H, S, D = in_3.shape
    out = torch.empty(B, H, S + 1, D, device=in_3.device, dtype=in_6.dtype)
    _rope_fwd_q16s256[(H, S + 1)](
        in_3, in_1, in_5, in_2, out,
        H, S,
        in_3.stride(1), in_3.stride(2),
        in_1.stride(0),
        out.stride(1), out.stride(2),
        in_2.stride(1),
        D=D,
    )
    return out


def pattern(in_3, in_1, in_5, in_2, in_6):
    tmp_1 = in_3 * in_1
    tmp_2 = in_3[..., 1::2]
    tmp_3 = -tmp_2
    tmp_4 = in_3[..., ::2]
    tmp_5 = torch.stack([tmp_3, tmp_4], -1)
    tmp_6 = tmp_5.reshape((1, 16, 256, 64))
    tmp_7 = tmp_6 * in_5
    tmp_8 = tmp_1 + tmp_7
    tmp_9 = torch.cat([in_2, tmp_8], dim=2)
    tmp_10 = tmp_9.type_as(in_6)
    return tmp_10


def replacement_args(in_3, in_1, in_5, in_2, in_6):
    return (in_3, in_1, in_5, in_2, in_6)


def replacement_func():
    return rope_q_fwd_16_256