import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.silu(tmp_6, inplace=True)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


@triton.jit
def fused_avgpool_bn_silu_kernel(
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    x_ptr,
    out_ptr,
    EPS: tl.constexpr,
):
    c = tl.program_id(0)

    rows = tl.arange(0, 8)[:, None]
    cols = tl.arange(0, 8)[None, :]

    # x is logically reshaped from [4, 128, 256] to contiguous [1, 512, 16, 16]
    # Each channel occupies 16 * 16 = 256 contiguous elements.
    base = c * 256 + rows * 32 + cols * 2

    x00 = tl.load(x_ptr + base).to(tl.float32)
    x01 = tl.load(x_ptr + base + 1).to(tl.float32)
    x10 = tl.load(x_ptr + base + 16).to(tl.float32)
    x11 = tl.load(x_ptr + base + 17).to(tl.float32)
    pooled = (x00 + x01 + x10 + x11) * 0.25

    mean = tl.load(running_mean_ptr + c).to(tl.float32)
    var = tl.load(running_var_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    normed = (pooled - mean) * tl.rsqrt(var + EPS)
    y = normed * weight + bias
    y = y / (1.0 + tl.exp(-y))

    out_offsets = c * 64 + rows * 8 + cols
    tl.store(out_ptr + out_offsets, y)


@torch.fx.wrap
def fused_reshape_avgpool_bn_silu(running_mean, running_var, bias, weight, x):
    out = torch.empty((1, 512, 8, 8), device=x.device, dtype=x.dtype)
    fused_avgpool_bn_silu_kernel[(512,)](
        running_mean,
        running_var,
        bias,
        weight,
        x,
        out,
        EPS=1e-5,
        num_warps=2,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_reshape_avgpool_bn_silu