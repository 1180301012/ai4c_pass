import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_2 = torch.nn.functional.unfold(x, kernel_size=(2, 2), stride=(2, 2))
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


def replacement_args(x):
    return (x,)


@triton.jit
def _unfold2x2_stride2_reshape_kernel(
    x_ptr,
    out_ptr,
    stride_c,
    stride_h,
    stride_w,
):
    pid = tl.program_id(0)
    c = pid // 4
    k = pid % 4
    kh = k // 2
    kw = k % 2

    l = tl.arange(0, 256)
    oh = l // 16
    ow = l % 16

    x_offsets = c * stride_c + (oh * 2 + kh) * stride_h + (ow * 2 + kw) * stride_w
    vals = tl.load(x_ptr + x_offsets)

    out_offsets = pid * 256 + l
    tl.store(out_ptr + out_offsets, vals)


@torch.fx.wrap
def _unfold2x2_stride2_reshape_triton(x):
    out = torch.empty((1, 128, 4, 256), device=x.device, dtype=x.dtype)
    grid = (512,)
    _unfold2x2_stride2_reshape_kernel[grid](
        x,
        out,
        x.stride(1),
        x.stride(2),
        x.stride(3),
    )
    return out


def replacement_func():
    return _unfold2x2_stride2_reshape_triton