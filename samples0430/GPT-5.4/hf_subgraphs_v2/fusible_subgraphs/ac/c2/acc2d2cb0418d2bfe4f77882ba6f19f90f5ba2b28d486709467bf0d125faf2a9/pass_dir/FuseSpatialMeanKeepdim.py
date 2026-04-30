import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches: x.mean((2, 3), keepdim=True)
def pattern(x):
    tmp_11 = x.mean((2, 3), keepdim=True)
    return tmp_11


# Argument extraction function
def replacement_args(x):
    return (x,)


@triton.jit
def _spatial_mean_keepdim_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    out_stride_n,
    out_stride_c,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    HW = H * W

    acc = 0.0
    for hw_start in tl.range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h = offs // W
        ww = offs % W
        x_offs = n * x_stride_n + c * x_stride_c + h * x_stride_h + ww * x_stride_w
        x = tl.load(x_ptr + x_offs, mask=mask, other=0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x, 0.0), axis=0)

    mean_val = acc / HW
    out_off = n * out_stride_n + c * out_stride_c
    tl.store(out_ptr + out_off, mean_val)


@torch.fx.wrap
def fused_spatial_mean_keepdim(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    _spatial_mean_keepdim_kernel[grid](
        x,
        out,
        N,
        C,
        H,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        out.stride(0),
        out.stride(1),
        BLOCK_HW=256,
    )
    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_spatial_mean_keepdim