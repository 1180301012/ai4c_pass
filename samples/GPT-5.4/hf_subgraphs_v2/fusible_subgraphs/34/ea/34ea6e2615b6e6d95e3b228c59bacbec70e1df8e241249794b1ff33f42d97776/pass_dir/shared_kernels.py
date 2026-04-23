import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 32}, num_warps=2),
        triton.Config({"BLOCK_HW": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
    ],
    key=["C", "H", "W", "OUT_DTYPE_ID"],
)
@triton.jit
def _dwconv_residual_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    C,
    H,
    W,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    OUT_DTYPE_ID: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    hw_block = tl.program_id(1)
    hw_start = hw_block * BLOCK_HW
    hw = hw_start + tl.arange(0, BLOCK_HW)
    mask = (c < C) & (hw < H * W)

    h = hw // W
    w = hw % W

    base = c * H * W

    acc = tl.load(b_ptr + c).to(tl.float32)
    acc = tl.broadcast_to(acc, [BLOCK_HW])
    residual = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for kh in range(3):
        ih = h + kh - 1
        hmask = (ih >= 0) & (ih < H)
        ih_safe = tl.where(ih < 0, 0, tl.where(ih >= H, H - 1, ih))
        for kw in range(3):
            iw = w + kw - 1
            wmask = (iw >= 0) & (iw < W)
            iw_safe = tl.where(iw < 0, 0, tl.where(iw >= W, W - 1, iw))
            cur_mask = mask & hmask & wmask
            inp = tl.load(x_ptr + base + ih_safe * W + iw_safe, mask=cur_mask, other=0.0).to(tl.float32)
            wt = tl.load(w_ptr + c * 9 + kh * 3 + kw).to(tl.float32)
            acc += inp * wt
            if kh == 1 and kw == 1:
                residual = inp

    acc += residual

    if OUT_DTYPE_ID == 0:
        out_val = acc.to(tl.float32)
    else:
        out_val = acc.to(tl.bfloat16)

    out_ptrs = out_ptr + c * out_stride_c + h * out_stride_h + w * out_stride_w
    tl.store(out_ptrs, out_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_SIZE": 1024}, num_warps=8),
    ],
    key=["C", "OUT_DTYPE_ID"],
)
@triton.jit
def _layernorm_kernel(
    x_ptr,
    y_ptr,
    weight_ptr,
    bias_ptr,
    rows,
    HW,
    C,
    x_stride_0,
    x_stride_1,
    x_stride_2,
    y_stride_0,
    y_stride_1,
    y_stride_2,
    eps,
    OUT_DTYPE_ID: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < C

    n_idx = row // HW
    hw_idx = row % HW

    x_ptrs = x_ptr + n_idx * x_stride_0 + hw_idx * x_stride_1 + offs * x_stride_2
    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / C
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / C
    rstd = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x_centered * rstd
    out = out * weight + bias

    if OUT_DTYPE_ID == 0:
        out_cast = out.to(tl.float32)
    else:
        out_cast = out.to(tl.bfloat16)

    y_ptrs = y_ptr + n_idx * y_stride_0 + hw_idx * y_stride_1 + offs * y_stride_2
    tl.store(y_ptrs, out_cast, mask=mask)


@torch.fx.wrap
def depthwise_conv_residual(x, weight, bias):
    _, c, h, w = x.shape
    out = torch.empty_like(x)

    if x.dtype == torch.float32:
        out_dtype_id = 0
    elif x.dtype == torch.bfloat16:
        out_dtype_id = 1
    else:
        raise RuntimeError(f"unsupported dtype: {x.dtype}")

    grid = (c, triton.cdiv(h * w, 64))
    _dwconv_residual_kernel[grid](
        x,
        weight,
        bias,
        out,
        c,
        h,
        w,
        out.stride(1),
        out.stride(2),
        out.stride(3),
        OUT_DTYPE_ID=out_dtype_id,
    )
    return out


@torch.fx.wrap
def layernorm_lastdim(x, weight, bias):
    n, hw, c = x.shape
    y = torch.empty((n, hw, c), device=x.device, dtype=x.dtype)

    if x.dtype == torch.float32:
        out_dtype_id = 0
    elif x.dtype == torch.bfloat16:
        out_dtype_id = 1
    else:
        raise RuntimeError(f"unsupported dtype: {x.dtype}")

    rows = n * hw
    _layernorm_kernel[(rows,)](
        x,
        y,
        weight,
        bias,
        rows,
        hw,
        c,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        y.stride(0),
        y.stride(1),
        y.stride(2),
        1e-5,
        OUT_DTYPE_ID=out_dtype_id,
    )
    return y


def replacement_dispatch(*args):
    route = args[-1]
    if route == "dwconv_residual":
        return depthwise_conv_residual(args[0], args[1], args[2])
    if route == "layernorm_lastdim":
        return layernorm_lastdim(args[0], args[1], args[2])
    raise RuntimeError(f"unknown route: {route}")