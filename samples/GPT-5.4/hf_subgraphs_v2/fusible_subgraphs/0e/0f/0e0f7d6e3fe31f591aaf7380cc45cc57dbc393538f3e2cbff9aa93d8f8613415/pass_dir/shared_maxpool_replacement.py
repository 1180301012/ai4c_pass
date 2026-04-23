import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_OW": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OW": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_OW": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_OW": 128}, num_warps=8, num_stages=2),
    ],
    key=["OW"],
)
@triton.jit
def _maxpool2d_3x3_s2_p1_kernel(
    x_ptr,
    y_ptr,
    N,
    C,
    H,
    W,
    OH,
    OW,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    y_stride_n,
    y_stride_c,
    y_stride_h,
    y_stride_w,
    BLOCK_OW: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_row = tl.program_id(1)

    ow = pid_w * BLOCK_OW + tl.arange(0, BLOCK_OW)
    ow_mask = ow < OW

    oh = pid_row % OH
    tmp = pid_row // OH
    c = tmp % C
    n = tmp // C

    ih_base = oh * 2 - 1
    iw_center = ow * 2

    neg_inf = -float("inf")
    maxv = tl.full((BLOCK_OW,), neg_inf, tl.float32)

    base_nc = x_ptr + n * x_stride_n + c * x_stride_c

    ih0 = ih_base + 0
    h0_mask = (ih0 >= 0) & (ih0 < H)
    row0 = base_nc + ih0 * x_stride_h
    m00 = ow_mask & h0_mask & (iw_center - 1 >= 0) & (iw_center - 1 < W)
    m01 = ow_mask & h0_mask & (iw_center + 0 >= 0) & (iw_center + 0 < W)
    m02 = ow_mask & h0_mask & (iw_center + 1 >= 0) & (iw_center + 1 < W)
    v00 = tl.load(row0 + (iw_center - 1) * x_stride_w, mask=m00, other=neg_inf)
    v01 = tl.load(row0 + (iw_center + 0) * x_stride_w, mask=m01, other=neg_inf)
    v02 = tl.load(row0 + (iw_center + 1) * x_stride_w, mask=m02, other=neg_inf)
    maxv = tl.maximum(maxv, v00.to(tl.float32))
    maxv = tl.maximum(maxv, v01.to(tl.float32))
    maxv = tl.maximum(maxv, v02.to(tl.float32))

    ih1 = ih_base + 1
    h1_mask = (ih1 >= 0) & (ih1 < H)
    row1 = base_nc + ih1 * x_stride_h
    m10 = ow_mask & h1_mask & (iw_center - 1 >= 0) & (iw_center - 1 < W)
    m11 = ow_mask & h1_mask & (iw_center + 0 >= 0) & (iw_center + 0 < W)
    m12 = ow_mask & h1_mask & (iw_center + 1 >= 0) & (iw_center + 1 < W)
    v10 = tl.load(row1 + (iw_center - 1) * x_stride_w, mask=m10, other=neg_inf)
    v11 = tl.load(row1 + (iw_center + 0) * x_stride_w, mask=m11, other=neg_inf)
    v12 = tl.load(row1 + (iw_center + 1) * x_stride_w, mask=m12, other=neg_inf)
    maxv = tl.maximum(maxv, v10.to(tl.float32))
    maxv = tl.maximum(maxv, v11.to(tl.float32))
    maxv = tl.maximum(maxv, v12.to(tl.float32))

    ih2 = ih_base + 2
    h2_mask = (ih2 >= 0) & (ih2 < H)
    row2 = base_nc + ih2 * x_stride_h
    m20 = ow_mask & h2_mask & (iw_center - 1 >= 0) & (iw_center - 1 < W)
    m21 = ow_mask & h2_mask & (iw_center + 0 >= 0) & (iw_center + 0 < W)
    m22 = ow_mask & h2_mask & (iw_center + 1 >= 0) & (iw_center + 1 < W)
    v20 = tl.load(row2 + (iw_center - 1) * x_stride_w, mask=m20, other=neg_inf)
    v21 = tl.load(row2 + (iw_center + 0) * x_stride_w, mask=m21, other=neg_inf)
    v22 = tl.load(row2 + (iw_center + 1) * x_stride_w, mask=m22, other=neg_inf)
    maxv = tl.maximum(maxv, v20.to(tl.float32))
    maxv = tl.maximum(maxv, v21.to(tl.float32))
    maxv = tl.maximum(maxv, v22.to(tl.float32))

    out_ptr = y_ptr + n * y_stride_n + c * y_stride_c + oh * y_stride_h + ow * y_stride_w
    tl.store(out_ptr, maxv, mask=ow_mask)


@torch.fx.wrap
def triton_max_pool2d_3x3_s2_p1(x):
    n, c, h, w = x.shape
    oh = (h + 1) // 2
    ow = (w + 1) // 2
    y = torch.empty((n, c, oh, ow), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(ow, meta["BLOCK_OW"]), n * c * oh)
    _maxpool2d_3x3_s2_p1_kernel[grid](
        x,
        y,
        n,
        c,
        h,
        w,
        oh,
        ow,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        y.stride(3),
    )
    return y