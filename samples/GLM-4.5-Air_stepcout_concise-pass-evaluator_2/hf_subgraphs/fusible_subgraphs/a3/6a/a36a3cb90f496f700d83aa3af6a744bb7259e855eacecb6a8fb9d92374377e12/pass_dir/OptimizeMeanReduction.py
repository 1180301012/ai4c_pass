import torch
import triton
import triton.language as tl

# Pattern matching function - must match exactly the computation in model.py
def pattern(tmp_6):
    # Mean operation exactly as in model.py - reduce over dimensions (2, 3) with keepdim=True
    tmp_7 = tmp_6.mean((2, 3), keepdim=True)
    return tmp_7

# Argument extraction function
def replacement_args(tmp_6):
    return (tmp_6,)

# Triton kernel for optimized mean reduction
@triton.jit
def optimized_mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    num_features,
    spatial_size,
):
    # Each program handles one feature in one batch element
    batch_idx = tl.program_id(0) // num_features
    feature_idx = tl.program_id(0) % num_features
    
    # Only process valid indices
    if batch_idx >= batch_size or feature_idx >= num_features:
        return
    
    # Calculate base offset for this batch and feature
    base_offset = batch_idx * num_features * spatial_size + feature_idx * spatial_size
    
    # Accumulate all spatial elements
    accumulator = 0.0
    for i in range(spatial_size):
        offset = base_offset + i
        accumulator += tl.load(x_ptr + offset)
    
    # Calculate mean by dividing by spatial size
    mean_value = accumulator / spatial_size
    
    # Store result at [batch_idx, feature_idx] position
    out_offset = batch_idx * num_features + feature_idx
    tl.store(out_ptr + out_offset, mean_value)

@torch.fx.wrap
def optimized_mean(tmp_6):
    # Get tensor dimensions
    batch_size, num_features, height, width = tmp_6.shape
    spatial_size = height * width
    
    # Choose optimal block and vector sizes
    BLOCK_SIZE = 1  # Each program handles one feature per batch
    VECTOR_SIZE = min(32, spatial_size)  # Vectorized loads, but bounded by spatial size
    
    # Calculate number of programs (total batch * num_features)
    num_programs = batch_size * num_features
    
    # Create output tensor with shape [batch_size, num_features, 1, 1]
    out = torch.empty(batch_size, num_features, 1, 1, device=tmp_6.device, dtype=tmp_6.dtype)
    
    # Launch the kernel
    optimized_mean_kernel[(num_programs,)](
        x_ptr=tmp_6,
        out_ptr=out,
        batch_size=batch_size,
        num_features=num_features,
        spatial_size=spatial_size
    )
    
    return out

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return optimized_mean