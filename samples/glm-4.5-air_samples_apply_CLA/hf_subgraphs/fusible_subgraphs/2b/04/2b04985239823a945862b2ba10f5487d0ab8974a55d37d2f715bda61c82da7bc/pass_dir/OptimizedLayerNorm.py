import torch
import triton
import triton.language as tl

def layer_norm_pattern(x, normalized_shape, weight, bias, eps):
    """Pattern matching for LayerNorm optimization"""
    return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    """Extract arguments for optimized LayerNorm kernel"""
    B, HW, C = x.shape
    return (x, weight, bias, eps, B, HW, C)

@triton.jit
def layernorm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    y_ptr,
    eps: tl.constexpr,
    B: tl.constexpr,
    HW: tl.constexpr,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized LayerNorm kernel using Triton"""
    pid = tl.program_id(0)
    
    # Process one sequence position at a time
    offsets = pid * C + tl.arange(0, C)
    mask = offsets < C
    
    # Load data for this position
    x = tl.load(x_ptr + pid * C * 4 + offsets * 4, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets * 4, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets * 4, mask=mask, other=0.0)
    
    # Compute mean
    mean = tl.sum(x) / C
    
    # Compute variance with numerical stability
    var = tl.sum((x - mean) * (x - mean)) / C
    std = tl.sqrt(var + eps)
    
    # Normalize and apply scale/bias
    y = (x - mean) / std * weight + bias
    
    # Store result
    tl.store(y_ptr + pid * C * 4 + offsets * 4, y, mask=mask)

@torch.fx.wrap
def optimized_layernorm(x, weight, bias, eps, B, HW, C):
    """Optimized LayerNorm kernel wrapper"""
    block_size = 1024
    num_programs = B * HW
    
    output = torch.empty_like(x)
    layernorm_kernel[(num_programs,)](
        x, weight, bias, output, eps, B, HW, C, block_size
    )
    return output

def replacement_func():
    return optimized_layernorm