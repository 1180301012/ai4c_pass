import torch
import triton
import triton.language as tl

def pattern(x, normalized_shape, weight, bias, eps):
    """Pattern matching for layer_norm operation"""
    # Pattern function should NOT call any torch operations
    # Just return input parameters to define the structure
    return (x, normalized_shape, weight, bias, eps)

def replacement_args(x, normalized_shape, weight, bias, eps):
    """Extract arguments for layer norm replacement"""
    return (x, normalized_shape, weight, bias, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    n_features: tl.constexpr,
    n_elements: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance layer norm implementation"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight and bias (broadcasting)
    weight = tl.load(weight_ptr + (offsets % n_features), mask=mask, other=0.0)
    bias = tl.load(bias_ptr + (offsets % n_features), mask=mask, other=0.0)
    
    # Apply layer normalization
    mean = tl.sum(x) / n_features
    var = tl.sum((x - mean) * (x - mean)) / n_features
    std = tl.sqrt(var + eps)
    
    # Normalize and apply weight/bias
    out = (x - mean) / std * weight + bias
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, normalized_shape, eps):
    """Wrapper for optimized layer norm"""
    # Move parameters to GPU if needed
    if weight.device.type == 'cpu':
        weight = weight.cuda()
    if bias.device.type == 'cpu':
        bias = bias.cuda()
        
    # Broadcast weight and bias to match input shape if needed
    if len(weight.shape) == 1:
        weight = weight.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
    if len(bias.shape) == 1:
        bias = bias.view(1, 1, -1).expand(x.shape[0], x.shape[1], -1)
    
    n_elements = x.numel()
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    out = torch.empty_like(x)
    
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_features=x.shape[-1],
        n_elements=n_elements,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Replacement function that returns the optimized layer norm"""
    return optimized_layer_norm