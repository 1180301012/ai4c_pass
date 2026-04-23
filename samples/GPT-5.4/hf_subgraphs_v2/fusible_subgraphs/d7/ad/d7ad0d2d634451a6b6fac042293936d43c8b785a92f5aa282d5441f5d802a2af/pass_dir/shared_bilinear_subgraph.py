import torch
import triton
import triton.language as tl


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


@torch.fx.wrap
def shared_face_parsing_linear_permute_reshape_interpolate(bias, weight, x):
    linear = x @ weight.t()
    if bias is not None:
        linear = linear + bias

    x4 = linear.permute(0, 2, 1).reshape(linear.shape[0], -1, 16, 16)

    n = x4.shape[0]
    c = x4.shape[1]
    h = x4.shape[2]
    w = x4.shape[3]

    if h != 16 or w != 16:
        raise RuntimeError(f"Expected input spatial size 16x16, got {(h, w)}")

    tmp = torch.empty((n, c, 128, 16), device=x4.device, dtype=x4.dtype)
    out = torch.empty((n, c, 128, 128), device=x4.device, dtype=x4.dtype)

    x_stride_n, x_stride_c, x_stride_h, x_stride_w = x4.stride()
    tmp_stride_n, tmp_stride_c, tmp_stride_h, tmp_stride_w = tmp.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    grid_v = (n * c, triton.cdiv(128, 32))
    _upsample_bilinear_v_kernel[grid_v](
        x4,
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