import torch
import triton
import triton.language as tl


# Match the FX placeholder order observed by the backend graph:
# input activation, running_mean, running_var, weight, bias.
def pattern(x, running_mean, running_var, weight, bias):
    tmp_4 = x.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return tmp_6


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias)


@triton.jit
def fused_pool_bn_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    EPS: tl.constexpr,
):
    c = tl.program_id(0)

    pos = tl.arange(0, 64)
    ox = pos & 7
    oy = pos >> 3

    in_base = c * 256 + (oy << 5) + (ox << 1)

    x0 = tl.load(x_ptr + in_base).to(tl.float32)
    x1 = tl.load(x_ptr + in_base + 1).to(tl.float32)
    x2 = tl.load(x_ptr + in_base + 16).to(tl.float32)
    x3 = tl.load(x_ptr + in_base + 17).to(tl.float32)
    pooled = (x0 + x1 + x2 + x3) * 0.25

    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)

    scale = weight * tl.rsqrt(var + EPS)
    shift = bias - mean * scale
    y = pooled * scale + shift

    out_offsets = c * 64 + pos
    tl.store(out_ptr + out_offsets, y)


@torch.fx.wrap
def fused_pool_bn(x, running_mean, running_var, weight, bias):
    out = torch.empty((1, 512, 8, 8), device=x.device, dtype=x.dtype)
    fused_pool_bn_kernel[(512,)](
        x,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        EPS=1e-5,
        num_warps=2,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_pool_bn