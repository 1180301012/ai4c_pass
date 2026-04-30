import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, bias, weight, residual):
    y = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    y = torch.nn.functional.leaky_relu(y, 0.01, True)
    y = y + residual
    return y


def replacement_args(x, running_mean, running_var, bias, weight, residual):
    return (x, running_mean, running_var, weight, bias, residual)


@triton.jit
def _bn_leaky_add_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    weight_ptr,
    bias_ptr,
    residual_ptr,
    out_ptr,
    channels,
    hw,
    eps,
    negative_slope,
    CHUNK: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    c = (pid_nc % channels).to(tl.int32)
    plane_base = pid_nc.to(tl.int64) * hw.to(tl.int64)

    mean = tl.load(running_mean_ptr + c).to(tl.float32)
    var = tl.load(running_var_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    inv_std = tl.rsqrt(var + eps)

    offs0 = tl.arange(0, CHUNK)

    base0 = plane_base + offs0.to(tl.int64)
    mask0 = offs0 < hw
    x0 = tl.load(x_ptr + base0, mask=mask0, other=0).to(tl.float32)
    r0 = tl.load(residual_ptr + base0, mask=mask0, other=0).to(tl.float32)
    y0 = (x0 - mean) * inv_std
    y0 = y0 * weight + bias
    y0 = tl.where(y0 >= 0, y0, y0 * negative_slope)
    y0 = y0 + r0
    tl.store(out_ptr + base0, y0, mask=mask0)

    offs1 = CHUNK + tl.arange(0, CHUNK)
    base1 = plane_base + offs1.to(tl.int64)
    mask1 = offs1 < hw
    x1 = tl.load(x_ptr + base1, mask=mask1, other=0).to(tl.float32)
    r1 = tl.load(residual_ptr + base1, mask=mask1, other=0).to(tl.float32)
    y1 = (x1 - mean) * inv_std
    y1 = y1 * weight + bias
    y1 = tl.where(y1 >= 0, y1, y1 * negative_slope)
    y1 = y1 + r1
    tl.store(out_ptr + base1, y1, mask=mask1)

    offs2 = 2 * CHUNK + tl.arange(0, CHUNK)
    base2 = plane_base + offs2.to(tl.int64)
    mask2 = offs2 < hw
    x2 = tl.load(x_ptr + base2, mask=mask2, other=0).to(tl.float32)
    r2 = tl.load(residual_ptr + base2, mask=mask2, other=0).to(tl.float32)
    y2 = (x2 - mean) * inv_std
    y2 = y2 * weight + bias
    y2 = tl.where(y2 >= 0, y2, y2 * negative_slope)
    y2 = y2 + r2
    tl.store(out_ptr + base2, y2, mask=mask2)

    offs3 = 3 * CHUNK + tl.arange(0, CHUNK)
    base3 = plane_base + offs3.to(tl.int64)
    mask3 = offs3 < hw
    x3 = tl.load(x_ptr + base3, mask=mask3, other=0).to(tl.float32)
    r3 = tl.load(residual_ptr + base3, mask=mask3, other=0).to(tl.float32)
    y3 = (x3 - mean) * inv_std
    y3 = y3 * weight + bias
    y3 = tl.where(y3 >= 0, y3, y3 * negative_slope)
    y3 = y3 + r3
    tl.store(out_ptr + base3, y3, mask=mask3)


@torch.fx.wrap
def fused_bn_leaky_add(x, running_mean, running_var, weight, bias, residual):
    out = torch.empty_like(x)
    channels = x.shape[1]
    total_nc = x.shape[0] * x.shape[1]
    _bn_leaky_add_kernel[(total_nc,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        residual,
        out,
        channels,
        x.shape[2] * x.shape[3],
        1e-05,
        0.01,
        CHUNK=1024,
        num_warps=8,
    )
    return out


def replacement_func():
    return fused_bn_leaky_add