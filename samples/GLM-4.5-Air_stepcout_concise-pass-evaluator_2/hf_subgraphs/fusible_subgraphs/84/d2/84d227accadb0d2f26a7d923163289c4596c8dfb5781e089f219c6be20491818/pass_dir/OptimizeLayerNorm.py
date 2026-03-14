import torch
import triton
import triton.language as tl
import math

def pattern(x, weight, bias, normalized_shape, eps):
    """Pattern for layer normalization with custom epsilon"""
    result = torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)
    return result, weight  # Return result and weight for observable values

def replacement_args(x, weight, bias, normalized_shape, eps):
    return (x, weight, bias, normalized_shape, eps)

@triton.jit
def layernorm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    batch_size, seq_len, hidden_size,
    eps,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    """Optimized layer normalization kernel using Triton"""
    pid = tl.program_id(0)
    
    # Only process if within batch bounds
    if pid >= batch_size * seq_len:
        return
        
    # Compute the starting offset for this element
    offset = pid * hidden_size
    
    # Load x values for this element
    x_vals = tl.load(x_ptr + offset, mask=offset < (batch_size * seq_len * hidden_size), other=0.0)
    
    # Compute mean
    x_sum = tl.sum(x_vals.to(tl.float32))
    mean = x_sum / hidden_size
    
    # Compute variance
    x_centered = x_vals - mean
    x_var = tl.sum(x_centered * x_centered.to(tl.float32))
    var = x_var / hidden_size + eps
    
    # Compute inverse std dev
    inv_std = 1.0 / tl.sqrt(var)
    
    # Apply normalization and load parameters
    normalized = (x_vals - mean) * inv_std
    weight_vals = tl.load(weight_ptr, mask=offset < hidden_size, other=1.0)
    bias_vals = tl.load(bias_ptr, mask=offset < hidden_size, other=0.0)
    
    # Apply linear transformation
    result = normalized * weight_vals + bias_vals
    
    # Store result
    tl.store(out_ptr + offset, result, mask=offset < (batch_size * seq_len * hidden_size))

@torch.fx.wrap
def optimized_layernorm(x, weight, bias, normalized_shape, eps):
    """Optimized layer normalization wrapper"""
    batch_size, seq_len = x.shape[0], x.shape[1]
    hidden_size = normalized_shape[0]
    
    # Optimize block sizes based on hidden size
    if hidden_size <= 32:
        BLOCK_N = hidden_size
    elif hidden_size <= 128:
        BLOCK_N = 64
    else:
        BLOCK_N = 128
        
    BLOCK_M = min(256, batch_size * seq_len)
    
    # Calculate grid size
    total_elements = batch_size * seq_len
    grid_size = (total_elements + BLOCK_M - 1) // BLOCK_M
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel
    layernorm_kernel[grid_size](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        out_ptr=out,
        batch_size=batch_size,
        seq_len=seq_len,
        hidden_size=hidden_size,
        eps=eps,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return out, weight

def replacement_func():
    return optimized_layernorm