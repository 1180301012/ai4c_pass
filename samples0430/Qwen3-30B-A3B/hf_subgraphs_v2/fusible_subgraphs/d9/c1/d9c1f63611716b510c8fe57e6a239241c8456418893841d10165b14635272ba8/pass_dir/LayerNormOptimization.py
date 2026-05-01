import torch
import triton
import triton.language as tl

def pattern(x, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, (256,), weight, bias, eps)

def replacement_args(x, weight, bias, eps):
    return (x, weight, bias, eps)

@triton.jit
def layernorm_kernel(x_ptr, weight_ptr, bias_ptr, output_ptr, eps):
    block_start = tl.program_id(0) * 256
    x = tl.load(x_ptr + block_start + tl.arange(0, 256), mask=tl.arange(0, 256) < 256)
    mean = tl.sum(x, axis=0) / 256
    x_centered = x - mean
    variance = tl.sum(x_centered * x_centered, axis=0) / 256
    x_normalized = x_centered * tl.rsqrt(variance + eps)
    weight = tl.load(weight_ptr + tl.arange(0, 256), mask=tl.arange(0, 256) < 256, other=0.0)
    bias = tl.load(bias_ptr + tl.arange(0, 256), mask=tl.arange(0, 256) < 256, other=0.0)
    output = x_normalized * weight + bias
    tl.store(output_ptr + block_start + tl.arange(0, 256), output, mask=tl.arange(0, 256) < 256)

@torch.fx.wrap
def layer_norm_wrapper(x, weight, bias, eps):
    num_blocks = x.numel() // 256
    output = torch.empty_like(x)
    layernorm_kernel[(num_blocks,)](x, weight, bias, output, eps)
    return output

def replacement_func():
    return layer_norm_wrapper