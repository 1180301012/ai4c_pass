import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_generic(
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
def rope_generic_wrapper(x, cos, sin, cls, ref):
    B, H, S, D = x.shape
    out = torch.empty(B, H, S + 1, D, device=x.device, dtype=ref.dtype)
    _rope_fwd_generic[(H, S + 1)](
        x, cos, sin, cls, out,
        H, S,
        x.stride(1), x.stride(2),
        cos.stride(0),
        out.stride(1), out.stride(2),
        cls.stride(1),
        D=D,
    )
    return out


def pattern(x, cos, sin, cls, ref):
    x1 = x * cos
    x_odd = x[..., 1::2]
    x_neg_odd = -x_odd
    x_even = x[..., ::2]
    stacked = torch.stack([x_neg_odd, x_even], -1)
    rotated = stacked.reshape(x.shape)
    rot_sin = rotated * sin
    result = x1 + rot_sin
    out = torch.cat([cls, result], dim=2)
    return out.type_as(ref)


def replacement_args(x, cos, sin, cls, ref):
    return (x, cos, sin, cls, ref)


def replacement_func():
    return rope_generic_wrapper