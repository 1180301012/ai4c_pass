import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1, 'BLOCK_HW': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 1, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 4, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 4, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 128}, num_warps=8),
    ],
    key=['C', 'H', 'W'],
)
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
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = nc < (N * C)
    n = nc // C
    c = nc % C
    HW = H * W

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for hw_start in tl.range(0, HW, BLOCK_HW):
        hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = hw < HW
        h = hw // W
        ww = hw % W
        ptrs = x_ptr + n[:, None] * x_stride_n + c[:, None] * x_stride_c + h[None, :] * x_stride_h + ww[None, :] * x_stride_w
        mask = mask_c[:, None] & mask_hw[None, :]
        x = tl.load(ptrs, mask=mask, other=0).to(tl.float32)
        acc += tl.sum(x, axis=1)

    mean_val = acc / HW
    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c
    tl.store(out_ptrs, mean_val, mask=mask_c)


@triton.jit
def _bn_silu_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
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
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    HW = H * W

    rm = tl.load(running_mean_ptr + c).to(tl.float32)
    rv = tl.load(running_var_ptr + c).to(tl.float32)
    b = tl.load(bias_ptr + c).to(tl.float32)
    w = tl.load(weight_ptr + c).to(tl.float32)
    inv_std = tl.rsqrt(rv + eps)

    for hw_start in tl.range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        h = offs // W
        ww = offs % W

        x_offs = n * x_stride_n + c * x_stride_c + h * x_stride_h + ww * x_stride_w
        x = tl.load(x_ptr + x_offs, mask=mask, other=0).to(tl.float32)
        y = (x - rm) * inv_std * w + b
        sig = 1.0 / (1.0 + tl.exp(-y))
        z = y * sig

        out_offs = n * out_stride_n + c * out_stride_c + h * out_stride_h + ww * out_stride_w
        tl.store(out_ptr + out_offs, z, mask=mask)


@triton.jit
def _silu_mean_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
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
    mean_stride_n,
    mean_stride_c,
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
        sig = 1.0 / (1.0 + tl.exp(-x))
        y = x * sig

        out_offs = n * out_stride_n + c * out_stride_c + h * out_stride_h + ww * out_stride_w
        tl.store(out_ptr + out_offs, y, mask=mask)
        acc += tl.sum(tl.where(mask, y, 0.0), axis=0)

    mean_val = acc / HW
    mean_off = n * mean_stride_n + c * mean_stride_c
    tl.store(mean_ptr + mean_off, mean_val)


@torch.fx.wrap
def shared_replacement_dispatch(*args):
    route = args[-1]

    if route == "mean_only":
        x = args[0]
        N, C, H, W = x.shape
        out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
        grid = lambda META: (triton.cdiv(N * C, META['BLOCK_C']),)
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
        )
        return out

    if route == "bn_silu":
        x, running_mean, running_var, bias, weight = args[:5]
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        grid = (N * C,)
        _bn_silu_kernel[grid](
            x,
            running_mean,
            running_var,
            bias,
            weight,
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
            out.stride(2),
            out.stride(3),
            1e-05,
            BLOCK_HW=256,
        )
        return out

    if route == "silu_mean":
        x = args[0]
        N, C, H, W = x.shape
        out = torch.empty_like(x)
        mean = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
        grid = (N * C,)
        _silu_mean_kernel[grid](
            x,
            out,
            mean,
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
            out.stride(2),
            out.stride(3),
            mean.stride(0),
            mean.stride(1),
            BLOCK_HW=256,
        )
        return (out, mean)

    # Unreachable fallback route branch to keep wrapper total
    x = args[0]
    return x