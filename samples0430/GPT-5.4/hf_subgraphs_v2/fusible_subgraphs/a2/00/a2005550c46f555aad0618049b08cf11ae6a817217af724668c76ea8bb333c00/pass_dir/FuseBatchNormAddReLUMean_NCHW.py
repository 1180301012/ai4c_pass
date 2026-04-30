import torch
import triton
import triton.language as tl


EPS = 1e-5


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    tmp_5 = in_5 + tmp_4
    tmp_6 = torch.nn.functional.relu(tmp_5, inplace=False)
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_HW": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_HW": 512}, num_warps=8, num_stages=2),
    ],
    key=["HW"],
)
@triton.jit

def fused_bn_add_relu_mean_kernel(
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    x_ptr,
    residual_ptr,
    out_ptr,
    out_mean_ptr,
    C,
    HW,
    EPSILON: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    n = pid // C

    rm = tl.load(running_mean_ptr + c).to(tl.float32)
    rv = tl.load(running_var_ptr + c).to(tl.float32)
    bias = tl.load(bias_ptr + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)

    scale = weight * tl.rsqrt(rv + EPSILON)
    shift = bias - rm * scale

    base = (n * C + c) * HW
    acc = tl.zeros((1,), dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW

        x = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)

        y = tl.maximum(x * scale + shift + residual, 0.0)
        tl.store(out_ptr + base + hw_offsets, y, mask=mask)
        acc += tl.sum(y, axis=0)

    mean = acc / HW
    mean_offset = pid + tl.arange(0, 1)
    tl.store(out_mean_ptr + mean_offset, mean)


@torch.fx.wrap
def fused_bn_add_relu_mean(running_mean, running_var, bias, weight, x, residual):
    n = x.shape[0]
    c = x.shape[1]
    h = x.shape[2]
    w = x.shape[3]
    hw = h * w

    out = torch.empty_like(x)
    out_mean = torch.empty((n, c, 1, 1), device=x.device, dtype=x.dtype)

    grid = (n * c,)
    fused_bn_add_relu_mean_kernel[grid](
        running_mean,
        running_var,
        bias,
        weight,
        x,
        residual,
        out,
        out_mean,
        c,
        hw,
        EPSILON=EPS,
    )
    return (out, out_mean)


def replacement_func():
    return fused_bn_add_relu_mean