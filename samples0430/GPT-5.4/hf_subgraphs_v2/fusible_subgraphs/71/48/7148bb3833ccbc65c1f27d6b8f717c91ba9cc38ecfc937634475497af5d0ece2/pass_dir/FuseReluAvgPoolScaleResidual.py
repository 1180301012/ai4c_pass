import torch
import triton
import triton.language as tl


# Pattern matching function
# Must mirror the original graph exactly, including observable outputs.
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.relu(in_2, inplace=False)
    tmp_3 = torch.nn.functional.avg_pool2d(tmp_2, 3, 1, 1, False, False, None)
    tmp_4 = tmp_3 - tmp_2
    tmp_5 = in_0.unsqueeze(-1)
    tmp_6 = tmp_5.unsqueeze(-1)
    tmp_7 = tmp_6 * tmp_4
    tmp_8 = tmp_2 + tmp_7
    tmp_9 = in_1.unsqueeze(-1)
    tmp_10 = tmp_9.unsqueeze(-1)
    return (tmp_8, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_relu_avgpool_scale_residual_kernel(
    x_ptr,
    scale_ptr,
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
    out_stride_h,
    out_stride_w,
    tiles_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    tile_id = tl.program_id(0)
    nc = tl.program_id(1)

    n = nc // C
    c = nc % C

    tile_h = tile_id // tiles_w
    tile_w = tile_id % tiles_w

    h0 = tile_h * BLOCK_H
    w0 = tile_w * BLOCK_W

    rows = h0 + tl.arange(0, BLOCK_H + 2) - 1
    cols = w0 + tl.arange(0, BLOCK_W + 2) - 1

    row_mask = (rows >= 0) & (rows < H)
    col_mask = (cols >= 0) & (cols < W)
    halo_mask = row_mask[:, None] & col_mask[None, :]

    x_base = x_ptr + n * x_stride_n + c * x_stride_c
    x_ptrs = x_base + rows[:, None] * x_stride_h + cols[None, :] * x_stride_w
    x_halo = tl.load(x_ptrs, mask=halo_mask, other=0.0)
    x_halo = tl.maximum(x_halo, 0)

    valid = halo_mask.to(tl.float32)

    s00 = x_halo[0:BLOCK_H, 0:BLOCK_W]
    s01 = x_halo[0:BLOCK_H, 1:BLOCK_W + 1]
    s02 = x_halo[0:BLOCK_H, 2:BLOCK_W + 2]
    s10 = x_halo[1:BLOCK_H + 1, 0:BLOCK_W]
    s11 = x_halo[1:BLOCK_H + 1, 1:BLOCK_W + 1]
    s12 = x_halo[1:BLOCK_H + 1, 2:BLOCK_W + 2]
    s20 = x_halo[2:BLOCK_H + 2, 0:BLOCK_W]
    s21 = x_halo[2:BLOCK_H + 2, 1:BLOCK_W + 1]
    s22 = x_halo[2:BLOCK_H + 2, 2:BLOCK_W + 2]

    v00 = valid[0:BLOCK_H, 0:BLOCK_W]
    v01 = valid[0:BLOCK_H, 1:BLOCK_W + 1]
    v02 = valid[0:BLOCK_H, 2:BLOCK_W + 2]
    v10 = valid[1:BLOCK_H + 1, 0:BLOCK_W]
    v11 = valid[1:BLOCK_H + 1, 1:BLOCK_W + 1]
    v12 = valid[1:BLOCK_H + 1, 2:BLOCK_W + 2]
    v20 = valid[2:BLOCK_H + 2, 0:BLOCK_W]
    v21 = valid[2:BLOCK_H + 2, 1:BLOCK_W + 1]
    v22 = valid[2:BLOCK_H + 2, 2:BLOCK_W + 2]

    sum_vals = (
        tl.cast(s00, tl.float32) + tl.cast(s01, tl.float32) + tl.cast(s02, tl.float32)
        + tl.cast(s10, tl.float32) + tl.cast(s11, tl.float32) + tl.cast(s12, tl.float32)
        + tl.cast(s20, tl.float32) + tl.cast(s21, tl.float32) + tl.cast(s22, tl.float32)
    )
    counts = v00 + v01 + v02 + v10 + v11 + v12 + v20 + v21 + v22
    avg_vals = sum_vals / counts

    center = tl.cast(s11, tl.float32)
    scale = tl.cast(tl.load(scale_ptr + c), tl.float32)
    out_vals = center + scale * (avg_vals - center)

    out_rows = h0 + tl.arange(0, BLOCK_H)
    out_cols = w0 + tl.arange(0, BLOCK_W)
    out_mask = (out_rows[:, None] < H) & (out_cols[None, :] < W)

    out_base = out_ptr + n * out_stride_n + c * out_stride_c
    out_ptrs = out_base + out_rows[:, None] * out_stride_h + out_cols[None, :] * out_stride_w
    tl.store(out_ptrs, out_vals, mask=out_mask)


@torch.fx.wrap
def fused_relu_avgpool_scale_residual(in_0, in_1, in_2):
    out = torch.empty_like(in_2)

    N, C, H, W = in_2.shape
    x_stride_n, x_stride_c, x_stride_h, x_stride_w = in_2.stride()
    out_stride_n, out_stride_c, out_stride_h, out_stride_w = out.stride()

    BLOCK_H = 8
    BLOCK_W = 32
    tiles_h = triton.cdiv(H, BLOCK_H)
    tiles_w = triton.cdiv(W, BLOCK_W)

    grid = (tiles_h * tiles_w, N * C)

    fused_relu_avgpool_scale_residual_kernel[grid](
        in_2,
        in_0,
        out,
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
        out_stride_h,
        out_stride_w,
        tiles_w,
        BLOCK_H=BLOCK_H,
        BLOCK_W=BLOCK_W,
        num_warps=4,
    )

    return out, in_1.unsqueeze(-1).unsqueeze(-1)


def replacement_func():
    return fused_relu_avgpool_scale_residual