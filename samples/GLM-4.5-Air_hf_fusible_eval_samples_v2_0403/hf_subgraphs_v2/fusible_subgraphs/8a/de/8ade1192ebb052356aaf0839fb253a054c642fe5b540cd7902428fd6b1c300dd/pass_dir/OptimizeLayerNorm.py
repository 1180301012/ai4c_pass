import torch
import triton
import triton.language as tl
import math

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    normalized_shape,
    eps: tl.constexpr,
    block_size: tl.constexpr,
):
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * block_size
    offsets = block_start + tl.arange(0, block_size)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight and bias
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Apply layer normalization
    # y = (x - mean) / sqrt(var + eps) * weight + bias
    # We compute this in steps for better numerical stability
    
    # Calculate mean
    mean = tl.sum(x, axis=0) / normalized_shape
    
    # Calculate variance
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / normalized_shape
    
    # Apply normalization
    inv_std = 1.0 / tl.sqrt(var + eps)
    y = (x_centered * inv_std) * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(x, weight, bias, normalized_shape, eps=1e-05):
    # Get tensor properties
    n_elements = x.numel()
    
    # Determine optimal block size based on hidden size
    hidden_size = normalized_shape[0]
    block_size = min(1024, hidden_size * 4)  # Adjust block size based on hidden size
    num_programs = (n_elements + block_size - 1) // block_size
    
    # Create output tensor with same properties as input
    out = torch.empty_like(x)
    
    # Launch kernel
    layer_norm_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        n_elements=n_elements,
        normalized_shape=normalized_shape,
        eps=eps,
        block_size=block_size,
    )
    
    return out

# Pattern matching for layer norm
def pattern(tmp_13, in_3, in_2, normalized_shape):
    # Match layer norm with exact parameters from the model
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, normalized_shape, in_3, in_2, 1e-05)
    return tmp_14

def replacement_args(tmp_13, in_3, in_2, normalized_shape):
    return (tmp_13, in_3, in_2, normalized_shape)

def replacement_func():
    return optimized_layer_norm