import torch
import triton
import triton.language as tl

def pattern(spatial_features, x_weights):
    """Match the multiplication operation in spatial attention"""
    result = spatial_features.mul(x_weights)
    return result

def replacement_args(spatial_features, x_weights):
    """Extract arguments for optimization"""
    return spatial_features, x_weights

@triton.jit
def multiplication_kernel(
    spatial_ptr, weights_ptr, out_ptr,
    spatial_shape, weights_shape, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized element-wise multiplication using Triton - handles different shapes"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Handle spatial indexing (works with flattened data)
    spatial_val = tl.load(spatial_ptr + offsets, mask=mask, other=0.0)
    
    # For weights, we need handle broadcasting properly
    if weights_shape[0] == 1 and weights_shape[1] == 1:  # [1, 1, 1, 64] pattern
        # Broadcast across spatial dimensions
        spatial_flat_idx = offsets
        spatial_dim_size = spatial_shape[-1]
        weight_idx = spatial_flat_idx % spatial_dim_size
        weight_val = tl.load(weights_ptr + weight_idx, mask=mask, other=0.0)
    elif weights_shape[0] == 1 and weights_shape[3] == 1:  # [1, 1, 64, 1] pattern  
        # Broadcast across spatial dimensions
        spatial_flat_idx = offsets
        spatial_h_size = spatial_shape[-2]
        spatial_w_size = spatial_shape[-1]
        spatial_grid_idx = spatial_flat_idx % (spatial_h_size * spatial_w_size)
        weight_idx = spatial_grid_idx // spatial_w_size  # Extract height dimension
        weight_val = tl.load(weights_ptr + weight_idx, mask=mask, other=0.0)
    else:
        # Fallback: try linear indexing (should match for same-sized tensors)
        weight_val = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
    
    result = spatial_val * weight_val
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_multiplication(spatial_features, x_weights):
    """Optimized multiplication using Triton - only allowed operations"""
    # Use only allowed operations: get total elements without flattening
    total_elements = spatial_features.numel()
    
    # Create output tensor with same shape and dtype
    output = torch.empty_like(spatial_features)
    
    # Get tensor shapes for proper indexing
    spatial_shape = spatial_features.shape
    weights_shape = x_weights.shape
    
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel with shape information
    multiplication_kernel[(num_programs,)](
        spatial_features, x_weights, output,
        spatial_shape, weights_shape, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Return result in the same shape (no reshape needed)
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_multiplication