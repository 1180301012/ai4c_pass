import torch
import triton
import triton.language as tl


# Simple pattern: just match the add + batch_norm part
# This is a simpler pattern to test if framework works
def pattern(x, mean, var, weight, bias):
    y = x + 1.0  # Simple add
    out = torch.nn.functional.batch_norm(y, mean, var, weight, bias, False, 0.1, 1e-05)
    return out


def replacement_args(x, mean, var, weight, bias):
    return (x, mean, var, weight, bias)


# Simple kernel for add + bn
@triton.jit
def add_bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    N: tl.constexpr, C: tl.constexpr,
    eps: tl.constexpr,
):
    # Simple element-wise add + bn
    pid = tl.program_id(0)
    offs = tl.arange(0, C)
    
    x = tl.load(x_ptr + offs, mask=offs < C, other=0.0)
    mean = tl.load(mean_ptr + offs, mask=offs < C, other=0.0)
    var = tl.load(var_ptr + offs, mask=offs < C, other=1.0)
    weight = tl.load(weight_ptr + offs, mask=offs < C, other=1.0)
    bias = tl.load(bias_ptr + offs, mask=offs < C, other=0.0)
    
    y = x + 1.0
    normalized = (y - mean) / tl.sqrt(var + eps)
    out = normalized * weight + bias
    
    tl.store(out_ptr + offs, out, mask=offs < C)


@torch.fx.wrap
def simple_add_bn_wrapper(x, mean, var, weight, bias):
    C = x.numel()
    out = torch.empty_like(x)
    
    add_bn_kernel[(1,)](
        x, mean, var, weight, bias, out,
        1, C, 1e-05
    )
    return out


def replacement_func():
    return simple_add_bn_wrapper