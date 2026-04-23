import torch
import triton
import triton.language as tl


# Pattern matching function
# Match the full subgraph so the structure exactly mirrors model.py.
def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    tmp_3 = linear.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(x.shape[0], -1, 16, 16)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    return tmp_5


# Argument extraction function
def replacement_args(bias, weight, x):
    return (bias, weight, x)


@triton.jit
def _upsample_bilinear_v_kernel(
    x_ptr,
    tmp_ptr,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    tmp_stride_n,
    tmp_stride_c,
    tmp_stride_h,
    tmp_stride_w,
    C,
    BLOCK_H: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)

    c = pid_nc % C
    n = pid_nc // C

    rows = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.arange(0, 16)

    # align_corners=False, in_h=16, out_h=128 => scale = 1/8
    src = rows.to(tl.float32) * 0.125 - 0.4375
    src_floor = tl.floor(src)

    src0 = tl.maximum(src_floor, 0)
    src0 = tl.minimum(src0, 15)
    src1 = tl.minimum(src0 + 1, 15)

    w1 = src - src_floor
    border = (src <= 0.0) | (src >= 15.0)
    w1 = tl.where(border, 0.0, w1)
    w0 = 1.0 - w1

    src0_i = src0.to(tl.int64)
    src1_i = src1.to(tl.int64)

    row_mask = rows < 128
    mask = row_mask[:, None]

    x_base = x_ptr + n * x_stride_n + c * x_stride_c
    tmp_base = tmp_ptr + n * tmp_stride_n + c * tmp_stride_c

    ptr0 = x_base + src0_i[:, None] * x_stride_h + cols[None, :] * x_stride_w
    ptr1 = x_base + src1_i[:, None] * x_stride_h + cols[None, :] * x_stride_w

    v0 = tl.load(ptr0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(ptr1, mask=mask, other=0.0).to(tl.float32)
    out = v0 * w0[:, None] + v1 * w1[:, None]

    out_ptrs = tmp_base + rows[:, None] * tmp_stride_h + cols[None, :] * tmp_stride_w
    tl.store(out_ptrs, out, mask=mask)


@triton.jit
def _upsample_bilinear_h_kernel(
    tmp_ptr,
    out_ptr,
    tmp_stride_n,
    tmp_stride_c,
    tmp_stride_h,
    tmp_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    C,
    BLOCK_H: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_h = tl.program_id(1)

    c = pid_nc % C
    n = pid_nc // C

    rows = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    cols = tl.arange(0, 128)

    # align_corners=False, in_w=16, out_w=128 => scale = 1/8
    src = cols.to(tl.float32) * 0.125 - 0.4375
    src_floor = tl.floor(src)

    src0 = tl.maximum(src_floor, 0)
    src0 = tl.minimum(src0, 15)
    src1 = tl.minimum(src0 + 1, 15)

    w1 = src - src_floor
    border = (src <= 0.0) | (src >= 15.0)
    w1 = tl.where(border, 0.0, w1)
    w0 = 1.0 - w1

    src0_i = src0.to(tl.int64)
    src1_i = src1.to(tl.int64)

    row_mask = rows < 128
    mask = row_mask[:, None]

    tmp_base = tmp_ptr + n * tmp_stride_n + c * tmp_stride_c
    out_base = out_ptr + n * out_stride_n + c * out_stride_c

    ptr0 = tmp_base + rows[:, None] * tmp_stride_h + src0_i[None, :] * tmp_stride_w
    ptr1 = tmp_base + rows[:, None] * tmp_stride_h + src1_i[None, :] * tmp_stride_w

    v0 = tl.load(ptr0, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(ptr1, mask=mask, other=0.0).to(tl.float32)
    out = v0 * w0[None, :] + v1 * w1[None, :]

    out_ptrs = out_base + rows[:, None] * out_stride_h + cols[None, :] * out_stride_w
    tl.store(out_ptrs, out, mask=mask)


# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def triton_bilinear_upsample_16_to_128(bias, weight, x):
    linear = x @ weight.t()
    if bias is not None:
        linear = linear + bias

    x = linear.permute(0, 2, 1).reshape(linear.shape[0], -1, 16, 16)

    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]

    if h != 16 or w != 16:
        raise RuntimeError(f"TritonBilinearUpsample16x16To128x128 expects input spatial size 16x16, got {(h, w)}")

    tmp = torch.empty((n, c, 128, 16), device=x.device, dtype=x.dtype)
    out = torch.empty((n, c, 128, 128), device=x.device, dtype=x.dtype)

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x.stride()
    tmp_stride_n, tmp_stride_c, tmp_stride_h, tmp_stride_w = tmp.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    grid_v = (n * c, triton.cdiv(128, 32))
    _upsample_bilinear_v_kernel[grid_v](
        x,
        tmp,
        x_stride_n,
        x_stride_c,
        x_stride_h,
        x_stride_w,
        tmp_stride_n,
        tmp_stride_c,
        tmp_stride_h,
        tmp_stride_w,
        c,
        BLOCK_H=32,
        num_warps=4,
        num_stages=2,
    )

    grid_h = (n * c, triton.cdiv(128, 8))
    _upsample_bilinear_h_kernel[grid_h](
        tmp,
        out,
        tmp_stride_n,
        tmp_stride_c,
        tmp_stride_h,
        tmp_stride_w,
        out_stride_n,
        out_stride_c,
        out_stride_h,
        out_stride_w,
        c,
        BLOCK_H=8,
        num_warps=4,
        num_stages=2,
    )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return triton_bilinear_upsample_16_to_128