import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_7):
    return torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)


def replacement_args(in_0, in_1, in_2, in_3, in_7):
    return (in_0, in_1, in_2, in_3, in_7)


@triton.jit
def _bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    B, C,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_C)
    mask = offs < C
    x = tl.load(x_ptr + row * C + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + offs, mask=mask, other=0.0)
    var = tl.load(var_ptr + offs, mask=mask, other=0.0)
    w = tl.load(weight_ptr + offs, mask=mask, other=0.0)
    b = tl.load(bias_ptr + offs, mask=mask, other=0.0)
    x_f = x.to(tl.float32)
    mean_f = mean.to(tl.float32)
    var_f = var.to(tl.float32)
    w_f = w.to(tl.float32)
    b_f = b.to(tl.float32)
    inv_std = tl.rsqrt(var_f + 1e-5)
    out_f = (x_f - mean_f) * inv_std * w_f + b_f
    tl.store(out_ptr + row * C + offs, out_f, mask=mask)


@torch.fx.wrap
def bn_triton(running_mean, running_var, bn_bias, bn_weight, x_bn):
    B = x_bn.shape[0]
    C = x_bn.shape[1]
    out = torch.empty_like(x_bn)
    _bn_kernel[(B,)](
        x_bn, running_mean, running_var, bn_weight, bn_bias, out,
        B, C,
        BLOCK_C=512,
    )
    return out


def replacement_func():
    return bn_triton