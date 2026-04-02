import torch
import triton
import triton.language as tl


@triton.jit
def _rope_fwd_k6s256(
    x_ptr, cos_ptr, sin_ptr, cls_ptr, out_ptr,
    H, S,
    stride_xh, stride_xs,
    stride_cs,
    stride_outh, stride_outs,
    stride_clsh,
    D: tl.constexpr,
):
    pid = tl.program_id(0)
    h = pid // (S + 1)
    s_out = pid % (S + 1)
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
def rope_k_fwd_6_256(in_0, in_4, in_6):
    S_pos = in_0.shape[0]
    D = in_0.shape[1] // 2
    B, H, S1, _ = in_4.shape

    sin = in_0[:, :D]
    cos = in_0[:, D:]
    k_cls = in_4[:, :, :1, :]
    k_seq = in_4[:, :, 1:, :]

    out = torch.empty(B, H, S1, D, device=in_4.device, dtype=in_6.dtype)
    _rope_fwd_k6s256[(H * S1,)](
        k_seq, cos, sin, k_cls, out,
        H, S_pos,
        k_seq.stride(1), k_seq.stride(2),
        cos.stride(0),
        out.stride(1), out.stride(2),
        k_cls.stride(1),
        D=D,
    )
    return out


def pattern(in_0, in_4, in_6):
    tmp_11 = in_4[:, :, :1, :]
    tmp_12 = in_4[:, :, 1:, :]
    tensor_split = in_0.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[..., 1::2]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[..., ::2]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, 6, 256, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(in_6)
    return tmp_25


def replacement_args(in_0, in_4, in_6):
    return (in_0, in_4, in_6)


def replacement_func():
    return rope_k_fwd_6_256