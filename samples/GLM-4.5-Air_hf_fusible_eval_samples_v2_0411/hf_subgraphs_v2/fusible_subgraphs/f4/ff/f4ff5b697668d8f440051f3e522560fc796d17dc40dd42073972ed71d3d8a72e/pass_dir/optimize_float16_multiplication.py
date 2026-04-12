import torch
import triton
import triton.language as tl

def pattern(spatial_features, y_weights):
    """Match multiplication operation with y_weights for float16 optimization"""
    result = spatial_features.mul(y_weights)
    return result

def replacement_args(spatial_features, y_weights):
    """Extract arguments for optimization"""
    return spatial_features, y_weights

@triton.jit
def fast_multiplication_kernel(
    spatial_ptr, weights_ptr, out_ptr,
    spatial_shape, weights_shape, total_elements,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized element-wise multiplication with better float16 performance"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load spatial data
    spatial_val = tl.load(spatial_ptr + offsets, mask=mask, other=0.0)
    
    # Optimized weight indexing for float16 patterns
    if weights_shape[0] == 1 and weights_shape[1] == 1:  # [1, 1, 1, 64] pattern
        spatial_flat_idx = offsets
        spatial_dim_size = spatial_shape[-1]
        weight_idx = spatial_flat_idx % spatial_dim_size
        weight_val = tl.load(weights_ptr + weight_idx, mask=mask, other=0.0)
    elif weights_shape[0] == 1 and weights_shape[3] == 1:  # [1, 1, 64, 1] pattern
        spatial_flat_idx = offsets
        spatial_h_size = spatial_shape[-2]
        spatial_w_size = spatial_shape[-1]
        spatial_grid_idx = spatial_flat_idx % (spatial_h_size * spatial_w_size)
        weight_idx = spatial_grid_idx // spatial_w_size
        weight_val = tl.load(weights_ptr + weight_idx, mask=mask, other=0.0)
    else:
        # Use more conservative indexing for other patterns
        if weights_shape[-1] == spatial_shape[-1]:  # Match last dimension
            weight_val = tl.load(weights_ptr + offsets, mask=mask, other=0.0)
        else:
            # Fallback pattern
            spatial_flat_idx = offsets
            spatial_total_elements = spatial_shape.numel()
            weight_idx = (spatial_flat_idx % spatial_total_elements) % weights_shape.numel()
            weight_val = tl.load(weights_ptr + weight_idx, mask=mask, other=0.0)
    
    # Perform multiplication with float16 optimization
    result = spatial_val * weight_val
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_float16_multiplication(spatial_features, y_weights):
    """Optimized multiplication for float16 performance"""
    total_elements = spatial_features.numel()
    output = torch.empty_like(spatial_features)
    
    spatial_shape = spatial_features.shape
    weights_shape = y_weights.shape
    
    # Use larger block size for float16 performance
    BLOCK_SIZE = 2048  # Increased for better GPU utilization
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fast_multiplication_kernel[(num_programs,)](
        spatial_features, y_weights, output,
        spatial_shape, weights_shape, total_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_float16_multiplication