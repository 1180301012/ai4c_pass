import torch
import triton
import triton.language as tl


def _pattern_impl(x, running_mean, running_var, bias, weight):
    tmp_0 = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp_1 = torch.nn.functional.relu(tmp_0, inplace=False)
    return tmp_1


pattern = torch.fx.symbolic_trace(_pattern_impl)


def replacement_args(x, running_mean, running_var, bias, weight):
    return (x, running_mean, running_var, bias, weight)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_bn_relu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    channels,
    hw,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    c = (offs // hw) % channels

    x = tl.load(x_ptr + offs, mask=mask, other=0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1).to(tl.float32)

    y = (x - mean) * tl.math.rsqrt(var + eps) * weight + bias
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_bn_relu(x, running_mean, running_var, bias, weight):
    out = torch.empty_like(x)
    n_elements = x.numel()
    channels = x.shape[1]
    hw = x.shape[2] * x.shape[3]
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    _fused_bn_relu_kernel[grid](
        x,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        n_elements,
        channels,
        hw,
        0.001,
    )
    return out


def replacement_func():
    return fused_bn_relu