import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias):
    bn = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu_out = torch.nn.functional.relu(bn, inplace=False)
    cast_out = relu_out.to(torch.bfloat16)
    return cast_out


def replacement_args(x, running_mean, running_var, weight, bias):
    return (x, running_mean, running_var, weight, bias, "bf16")


@triton.jit
def bn_relu_f16_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    mean_val = tl.load(mean_ptr + c).to(tl.float32)
    var_val  = tl.load(var_ptr  + c).to(tl.float32)
    w_val    = tl.load(weight_ptr + c).to(tl.float32)
    b_val    = tl.load(bias_ptr   + c).to(tl.float32)
    inv_std  = 1.0 / tl.sqrt(var_val + eps)
    base     = pid * HW
    offsets  = tl.arange(0, BLOCK_HW)
    mask     = offsets < HW
    x_val    = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    y        = (x_val - mean_val) * inv_std * w_val + b_val
    y        = tl.where(y > 0.0, y, 0.0)
    tl.store(out_ptr + base + offsets, y.to(tl.float16), mask=mask)


@triton.jit
def bn_relu_bf16_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    c = pid % C
    mean_val = tl.load(mean_ptr + c).to(tl.float32)
    var_val  = tl.load(var_ptr  + c).to(tl.float32)
    w_val    = tl.load(weight_ptr + c).to(tl.float32)
    b_val    = tl.load(bias_ptr   + c).to(tl.float32)
    inv_std  = 1.0 / tl.sqrt(var_val + eps)
    base     = pid * HW
    offsets  = tl.arange(0, BLOCK_HW)
    mask     = offsets < HW
    x_val    = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    y        = (x_val - mean_val) * inv_std * w_val + b_val
    y        = tl.where(y > 0.0, y, 0.0)
    tl.store(out_ptr + base + offsets, y.to(tl.bfloat16), mask=mask)


@torch.fx.wrap
def bn_relu_cast_dispatch(x, running_mean, running_var, weight, bias, route):
    N, C, H, W = x.shape
    HW = H * W
    BLOCK_HW = triton.next_power_of_2(HW)
    grid = (N * C,)
    if route == "f16":
        out = torch.empty(N, C, H, W, dtype=torch.float16, device=x.device)
        if not x.is_cuda:
            return out
        bn_relu_f16_kernel[grid](
            x, running_mean, running_var, weight, bias, out,
            C, HW, eps=1e-05, BLOCK_HW=BLOCK_HW,
        )
    else:
        out = torch.empty(N, C, H, W, dtype=torch.bfloat16, device=x.device)
        if not x.is_cuda:
            return out
        bn_relu_bf16_kernel[grid](
            x, running_mean, running_var, weight, bias, out,
            C, HW, eps=1e-05, BLOCK_HW=BLOCK_HW,
        )
    return out


def replacement_func():
    return bn_relu_cast_dispatch