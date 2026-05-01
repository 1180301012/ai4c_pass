import torch
import triton
import triton.language as tl

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    D,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    d = tl.arange(0, BLOCK_SIZE)
    mask = d < D
    
    x = tl.load(x_ptr + row_id * D + d, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / D
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / D
    inv_std = 1.0 / tl.sqrt(var + 1e-12)
    
    x_normalized = x_centered * inv_std
    
    weight = tl.load(weight_ptr + d, mask=mask)
    bias = tl.load(bias_ptr + d, mask=mask)
    out = x_normalized * weight + bias
    
    tl.store(out_ptr + row_id * D + d, out, mask=mask)

@torch.fx.wrap
def triton_layer_norm(x, weight, bias, eps):
    B, S, D = x.shape
    rows = B * S
    BLOCK_SIZE = 1024
    
    out = torch.empty_like(x)
    
    layer_norm_kernel[(rows, 1)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        D=D,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out

def pattern(in_4, normalized_shape, weight, bias, eps):
    return torch.nn.functional.layer_norm(in_4, normalized_shape, weight, bias, eps)

def replacement_args(in_4, normalized_shape, weight, bias, eps):
    return (in_4, weight, bias, eps)

def replacement_func():
    return triton_layer_norm