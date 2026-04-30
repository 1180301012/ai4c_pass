import torch
import triton
import triton.language as tl


def pattern(x, in_1, in_0):
    tmp_3 = torch.rand([])
    tmp_4 = torch.nn.functional.layer_norm(x, (1024,), in_1, in_0, 1e-05)
    return tmp_4


def replacement_args(x, in_1, in_0):
    return (x, in_1, in_0)


@triton.jit
def layernorm_1024_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    eps,
):
    row = tl.program_id(0)
    cols = tl.arange(0, 1024)
    offsets = row * 1024 + cols

    x = tl.load(x_ptr + offsets)
    x_fp32 = x.to(tl.float32)
    mean = tl.sum(x_fp32, axis=0) * (1.0 / 1024.0)
    diff = x_fp32 - mean
    var = tl.sum(diff * diff, axis=0) * (1.0 / 1024.0)
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + cols).to(tl.float32)
    bias = tl.load(bias_ptr + cols).to(tl.float32)
    out = diff * inv_std * weight + bias
    tl.store(out_ptr + offsets, out)


@torch.fx.wrap
def layernorm_1024_triton(x, weight, bias):
    out = torch.empty_like(x)
    rows = x.numel() // 1024
    layernorm_1024_kernel[(rows,)](x, weight, bias, out, 1e-5)
    return out


def replacement_func():
    return layernorm_1024_triton