import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr, 
    bias_ptr,
    out_ptr,
    batch_size: tl.constexpr,
    eps: tl.constexpr,
):
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    idx = pid * 256
    offsets = idx + tl.arange(0, 256)
    
    x = tl.load(x_ptr + offsets)
    
    mean = tl.sum(x) / 256.0
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered) / 256.0
    std = tl.sqrt(var + eps)
    
    weight = tl.load(weight_ptr)
    bias = tl.load(bias_ptr)
    
    out = x_centered / std * weight + bias
    
    tl.store(out_ptr + offsets, out)

@torch.fx.wrap
def triton_layer_norm(x, normalized_shape, weight, bias, eps):
    N = x.numel()
    normalized_size = normalized_shape[0]
    batch_size = N // normalized_size
    
    out = torch.empty_like(x)
    
    layer_norm_kernel[(batch_size,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        eps=eps,
    )
    
    return out

def replacement_func():
    return triton_layer_norm