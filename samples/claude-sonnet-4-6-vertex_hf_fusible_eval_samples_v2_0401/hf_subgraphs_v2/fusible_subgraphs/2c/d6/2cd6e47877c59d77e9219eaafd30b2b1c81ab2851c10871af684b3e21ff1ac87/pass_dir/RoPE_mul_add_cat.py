import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mac_kernel(
    x_ptr, cos_ptr, rot_ptr, sin_ptr, cls_ptr, out_ptr,
    H, S,
    stride_xh, stride_xs,
    stride_coss,
    stride_roth, stride_rots,
    stride_sins,
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
        xv = tl.load(x_ptr + h * stride_xh + s * stride_xs + d)
        cv = tl.load(cos_ptr + s * stride_coss + d)
        rv = tl.load(rot_ptr + h * stride_roth + s * stride_rots + d)
        sv = tl.load(sin_ptr + s * stride_sins + d)
        tl.store(out_ptr + out_base + d, xv * cv + rv * sv)


@torch.fx.wrap
def fused_mul_add_cat_typecast(x, cos, rotate_result, sin_emb, cls, ref):
    B, H, S, D = rotate_result.shape
    out = torch.empty(B, H, S + 1, D, device=x.device, dtype=ref.dtype)
    _fused_mac_kernel[(H, S + 1)](
        x, cos, rotate_result, sin_emb, cls, out,
        H, S,
        x.stride(1), x.stride(2),
        cos.stride(0),
        rotate_result.stride(1), rotate_result.stride(2),
        sin_emb.stride(0),
        out.stride(1), out.stride(2),
        cls.stride(1),
        D=D,
    )
    return out


def pattern(x, cos, rotate_result, sin_emb, cls, ref):
    x_cos = x * cos
    rot_sin = rotate_result * sin_emb
    result = x_cos + rot_sin
    out = torch.cat([cls, result], dim=2)
    return out.type_as(ref)


def replacement_args(x, cos, rotate_result, sin_emb, cls, ref):
    return (x, cos, rotate_result, sin_emb, cls, ref)


def replacement_func():
    return fused_mul_add_cat_typecast