import torch
import triton
import triton.language as tl

def pattern(tmp_8, in_1, in_0, feature_dim, eps=1e-05):
    """
    Pattern matching: layer normalization operation
    Matches torch.nn.functional.layer_norm(tmp_8, (feature_dim,), in_1, in_0, eps)
    """
    return torch.nn.functional.layer_norm(tmp_8, (feature_dim,), in_1, in_0, eps)

def replacement_args(tmp_8, in_1, in_0, feature_dim, eps=1e-05):
    return (tmp_8, in_1, in_0, feature_dim, eps)

@triton.jit
def layer_norm_kernel(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements_x,
    n_elements_weight,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized layer normalization kernel using Triton
    """
    # Each program handles a contiguous block of data for one feature dimension
    pid = tl.program_id(0)
    n_features = tl.cdiv(n_elements_weight, BLOCK_SIZE)
    
    if pid >= n_features:
        return
    
    # Process one feature at a time for better memory coalescing
    feature_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    feature_mask = feature_idx < n_elements_weight
    
    # Get the weight and bias for this feature
    weight = tl.load(weight_ptr + feature_idx, mask=feature_mask, other=1.0)
    bias = tl.load(bias_ptr + feature_idx, mask=feature_mask, other=0.0)
    
    # Mean and variance calculation using parallel reduction
    # We'll calculate mean and variance for each sequence position across features
    block_start_x = tl.program_id(1) * BLOCK_SIZE
    offsets_x = block_start_x + tl.arange(0, BLOCK_SIZE)
    mask_x = offsets_x < n_elements_x
    
    # Load input data for one block
    x_block = tl.load(x_ptr + offsets_x, mask=mask_x, other=0.0)
    
    # Calculate mean for this block
    block_mean = tl.sum(x_block, axis=0) / tl.sum(mask_x)
    block_var = tl.sum((x_block - block_mean) ** 2, axis=0) / tl.sum(mask_x)
    
    # Standard normalization
    inv_std = tl.rsqrt(block_var + eps)
    
    # Normalize and apply weight/bias
    y = (x_block - block_mean) * inv_std * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets_x, y, mask=mask_x)

@triton.jit
def layer_norm_kernel_simple(
    x_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    weight_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple but effective layer normalization kernel
    """
    # Each program processes a block of elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Load weight and bias for each element in the block
    weight_offsets = offsets % weight_size
    bias_offsets = offsets % weight_size
    
    weight = tl.load(weight_ptr + weight_offsets, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + bias_offsets, mask=mask, other=0.0)
    
    # Calculate mean and variance for this block
    x_sum = tl.sum(x)
    x_sq_sum = tl.sum(x * x)
    count = tl.sum(mask)
    
    block_mean = x_sum / count
    block_var = (x_sq_sum / count) - (block_mean * block_mean)
    
    # Ensure variance is positive
    block_var = tl.maximum(block_var, 0.0)
    
    # Calculate inverse standard deviation
    inv_std = tl.rsqrt(block_var + eps)
    
    # Apply normalization: (x - mean) / std * weight + bias  
    normalized = (x - block_mean) * inv_std
    y = normalized * weight + bias
    
    # Store result
    tl.store(out_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def optimized_layer_norm(tmp_8, in_1, in_0, feature_dim, eps=1e-05):
    """
    Optimized layer normalization using Triton
    """
    n_elements = tmp_8.numel()
    weight_size = in_1.numel()
    
    # Allocate output
    out = torch.empty_like(tmp_8)
    
    # Use simple block-based processing
    BLOCK_SIZE = 1024
    
    # Calculate grid dimensions
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch optimized kernel
    layer_norm_kernel_simple[grid](
        tmp_8,
        in_1,
        in_0,
        out,
        n_elements,
        weight_size,
        eps,
        BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_layer_norm