import torch
import triton
import triton.language as tl


@triton.jit
def _depthwise_conv2d_mean_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    sum_ptr,
    n_batches,
    channels,
    h_in,
    w_in,
    h_out,
    w_out,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    w_stride_c,
    w_stride_h,
    w_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    STRIDE: tl.constexpr,
):
    pid_w = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_nc = tl.program_id(2)

    n = pid_nc // channels
    c = pid_nc - n * channels

    oh = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    ow = pid_w * BLOCK_W + tl.arange(0, BLOCK_W)
    out_mask = (oh[:, None] < h_out) & (ow[None, :] < w_out)

    base_x = x_ptr + n * x_stride_n + c * x_stride_c
    base_w = w_ptr + c * w_stride_c

    acc = tl.zeros((BLOCK_H, BLOCK_W), dtype=tl.float32)

    w00 = tl.load(base_w + 0 * w_stride_h + 0 * w_stride_w)
    w01 = tl.load(base_w + 0 * w_stride_h + 1 * w_stride_w)
    w02 = tl.load(base_w + 0 * w_stride_h + 2 * w_stride_w)
    w10 = tl.load(base_w + 1 * w_stride_h + 0 * w_stride_w)
    w11 = tl.load(base_w + 1 * w_stride_h + 1 * w_stride_w)
    w12 = tl.load(base_w + 1 * w_stride_h + 2 * w_stride_w)
    w20 = tl.load(base_w + 2 * w_stride_h + 0 * w_stride_w)
    w21 = tl.load(base_w + 2 * w_stride_h + 1 * w_stride_w)
    w22 = tl.load(base_w + 2 * w_stride_h + 2 * w_stride_w)

    ih0 = oh * STRIDE - 1
    iw0 = ow * STRIDE - 1

    ih = ih0 + 0
    iw = iw0 + 0
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w00.to(tl.float32)

    ih = ih0 + 0
    iw = iw0 + 1
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w01.to(tl.float32)

    ih = ih0 + 0
    iw = iw0 + 2
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w02.to(tl.float32)

    ih = ih0 + 1
    iw = iw0 + 0
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w10.to(tl.float32)

    ih = ih0 + 1
    iw = iw0 + 1
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w11.to(tl.float32)

    ih = ih0 + 1
    iw = iw0 + 2
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w12.to(tl.float32)

    ih = ih0 + 2
    iw = iw0 + 0
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w20.to(tl.float32)

    ih = ih0 + 2
    iw = iw0 + 1
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w21.to(tl.float32)

    ih = ih0 + 2
    iw = iw0 + 2
    mask = out_mask & (ih[:, None] >= 0) & (ih[:, None] < h_in) & (iw[None, :] >= 0) & (iw[None, :] < w_in)
    acc += tl.load(base_x + ih[:, None] * x_stride_h + iw[None, :] * x_stride_w, mask=mask, other=0.0).to(tl.float32) * w22.to(tl.float32)

    acc = tl.where(out_mask, acc, 0.0)

    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + oh[:, None] * out_stride_h + ow[None, :] * out_stride_w
    tl.store(out_ptrs, acc, mask=out_mask)

    stored = tl.load(out_ptrs, mask=out_mask, other=0.0).to(tl.float32)
    tile_sum = tl.sum(tl.sum(stored, axis=1), axis=0)
    tl.atomic_add(sum_ptr + pid_nc, tile_sum)


@triton.jit
def _finalize_mean_kernel(sum_ptr, mean_ptr, num_nc, inv_hw, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < num_nc
    vals = tl.load(sum_ptr + offs, mask=mask, other=0.0)
    vals = vals * inv_hw
    tl.store(mean_ptr + offs, vals, mask=mask)


def _run_depthwise_conv_mean(weight, x, stride):
    if stride == 1:
        block_h = 8
        block_w = 32
        num_warps = 4
    elif stride == 2:
        block_h = 8
        block_w = 32
        num_warps = 4
    else:
        raise RuntimeError(f"unsupported stride route: {stride}")

    n = x.shape[0]
    c = x.shape[1]
    h_in = x.shape[2]
    w_in = x.shape[3]
    h_out = (h_in + 2 - 3) // stride + 1
    w_out = (w_in + 2 - 3) // stride + 1

    out = torch.empty((n, c, h_out, w_out), device=x.device, dtype=x.dtype)
    sum_buf = torch.zeros((n * c,), device=x.device, dtype=torch.float32)
    mean_out = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)

    grid = (triton.cdiv(w_out, block_w), triton.cdiv(h_out, block_h), n * c)
    _depthwise_conv2d_mean_kernel[grid](
        x,
        weight,
        out,
        sum_buf,
        n,
        c,
        h_in,
        w_in,
        h_out,
        w_out,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        weight.stride(0),
        weight.stride(2),
        weight.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        BLOCK_H=block_h,
        BLOCK_W=block_w,
        STRIDE=stride,
        num_warps=num_warps,
        num_stages=2,
    )

    nc = n * c
    block = 256
    _finalize_mean_kernel[(triton.cdiv(nc, block),)](
        sum_buf,
        mean_out,
        nc,
        1.0 / (h_out * w_out),
        BLOCK=block,
        num_warps=4,
        num_stages=2,
    )
    return out, mean_out


@torch.fx.wrap
def dispatch_depthwise_conv_mean(weight, x, route):
    if route == "s1":
        return _run_depthwise_conv_mean(weight, x, 1)
    if route == "s2":
        return _run_depthwise_conv_mean(weight, x, 2)
    raise RuntimeError(f"unknown route: {route}")