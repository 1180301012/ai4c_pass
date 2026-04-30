import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches: batch_norm(inference) -> silu(inplace=True) -> mean((2, 3), keepdim=True)
def pattern(x, running_mean, running_var, bias, weight):
    tmp_9 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    return (tmp_10, tmp_11)


# Argument extraction function
def replacement_args(x, running_mean, running_var, bias, weight):
    return (x, running_mean, running_var, bias, weight)


@triton.jit
def _bn_silu_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
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

    acc = 0.0

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

        acc += tl.sum(tl.where(mask, z, 0.0), axis=0)

    mean_val = acc / HW
    mean_off = n * mean_stride_n + c * mean_stride_c
    tl.store(mean_ptr + mean_off, mean_val)


@torch.fx.wrap
def fused_batch_norm_silu_mean(x, running_mean, running_var, bias, weight):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    mean = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    grid = (N * C,)
    _bn_silu_mean_kernel[grid](
        x,
        running_mean,
        running_var,
        bias,
        weight,
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
        1e-05,
        BLOCK_HW=256,
    )
    return (out, mean)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_batch_norm_silu_mean