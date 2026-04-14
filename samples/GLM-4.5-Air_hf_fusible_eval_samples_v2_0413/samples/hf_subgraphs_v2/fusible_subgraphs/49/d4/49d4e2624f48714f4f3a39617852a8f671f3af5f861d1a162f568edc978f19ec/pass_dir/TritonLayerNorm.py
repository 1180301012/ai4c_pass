import torch
import triton
import triton.language as tl
import math

# Pattern matching function - matches the layer_norm call
def pattern(tmp_3, in_1, in_0):
    # Note: The normalized_shape and eps are fixed in the original call
    tmp_4 = torch.nn.functional.layer_norm(tmp_3, (768,), in_1, in_0, 1e-12)
    return tmp_4

# Argument extraction function
def replacement_args(tmp_3, in_1, in_0):
    return (tmp_3, in_1, in_0)

@triton.jit
def _layer_norm_var_mean_kernel(
    x_ptr,
    n_elements,
    var_ptr,
    mean_ptr,
    BLOCK_SIZE: tl.constexpr,
    REDUCE_BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each block handles one full vector (768 elements for layer norm)
    # We compute mean and variance for the entire vector
    if pid == 0:  # Only one block per vector since we reduce across 768 elements
        # Load entire vector
        offsets = tl.arange(0, n_elements)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        
        # Compute mean
        vector_mean = tl.sum(x) / n_elements
        
        # Compute variance: E[x^2] - (E[x])^2
        x_squared = x * x
        mean_squared = tl.sum(x_squared) / n_elements
        vector_var = mean_squared - vector_mean * vector_mean
        
        # Handle numerical stability
        vector_var = tl.maximum(vector_var, 1e-12)
        
        # Store results
        tl.store(mean_ptr, vector_mean)
        tl.store(var_ptr, vector_var)

@triton.jit
def _layer_norm_apply_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    var_ptr,
    mean_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + offsets, mask=mask, other=0.0)
    
    # Load mean and variance (scalar values)
    mean = tl.load(mean_ptr)
    var = tl.load(var_ptr)
    eps = 1e-12
    
    # Apply layer normalization: (x - mean) / sqrt(var + eps) * weight + bias
    normalized = (x - mean) * tl.rsqrt(var + eps)
    out = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap 
def triton_layer_norm(x, weight, bias):
    # Get tensor shapes
    n_elements = x.numel()
    hidden_size = weight.numel()  # Should be 768
    
    # Allocate buffers for mean and variance
    var_buffer = torch.empty((1,), dtype=torch.float32, device=x.device)
    mean_buffer = torch.empty((1,), dtype=torch.float32, device=x.device)
    
    # Compute mean and variance
    _layer_norm_var_mean_kernel[(1,)](
        x_ptr=x,
        n_elements=hidden_size,
        var_ptr=var_buffer,
        mean_ptr=mean_buffer,
        BLOCK_SIZE=1024,
        REDUCE_BLOCK_SIZE=32,
    )
    
    # Apply layer normalization
    out = torch.empty_like(x)
    
    # Calculate grid size
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    _layer_norm_apply_kernel[(num_programs,)](
        x_ptr=x,
        weight_ptr=weight,
        bias_ptr=bias,
        var_ptr=var_buffer,
        mean_ptr=mean_buffer,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns the kernel wrapper)
def replacement_func():
    return triton_layer_norm