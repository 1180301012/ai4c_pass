import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 256, 16, 16)
    tmp_5 = torch.nn.functional.batch_norm(tmp_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_6 = torch.nn.functional.silu(tmp_5, inplace=True)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_4, in_0, in_1, in_2, in_3, "reshape_256")


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64}, num_warps=1),
        triton.Config({"BLOCK_HW": 128}, num_warps=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
    ],
    key=["HW"],
)
@triton.jit

def _reshape_bn_silu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    d0,
    d1,
    d2,
    s0,
    s1,
    s2,
    HW,
    EPS: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    c = tl.program_id(0)
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW

    flat = c * HW + offs
    d1d2 = d1 * d2
    i0 = flat // d1d2
    rem = flat % d1d2
    i1 = rem // d2
    i2 = rem % d2
    x_offsets = i0 * s0 + i1 * s1 + i2 * s2

    x = tl.load(x_ptr + x_offsets, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + c).to(tl.float32)
    var = tl.load(var_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    y = (x - mean) * tl.rsqrt(var + EPS)
    y = y * weight + bias
    y = y * tl.sigmoid(y)

    tl.store(out_ptr + flat, y, mask=mask)


@torch.fx.wrap
def fused_reshape_bn_silu(x, running_mean, running_var, bias, weight, route):
    c = running_mean.numel()
    if route == "reshape_256":
        h = 16
        w = 16
        hw = 256
    elif route == "reshape_64":
        h = 8
        w = 8
        hw = 64
    else:
        hw = x.numel() // c
        h = int(hw ** 0.5)
        w = hw // h

    out = torch.empty((1, c, h, w), device=x.device, dtype=x.dtype)

    grid = (c,)
    _reshape_bn_silu_kernel[grid](
        x,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        x.shape[0],
        x.shape[1],
        x.shape[2],
        x.stride(0),
        x.stride(1),
        x.stride(2),
        hw,
        EPS=1e-5,
    )
    return out


def replacement_func():
    return fused_reshape_bn_silu