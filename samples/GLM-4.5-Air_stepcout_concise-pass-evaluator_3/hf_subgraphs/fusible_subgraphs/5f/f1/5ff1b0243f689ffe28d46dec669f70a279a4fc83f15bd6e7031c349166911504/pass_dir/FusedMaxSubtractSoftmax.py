import torch
import triton
import triton.language as tl

# Pattern matching function - matches the max operation
def pattern(in_0):
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    return tmp_0[0]

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

@triton.jit
def max_along_dim_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_features,
    n_positions,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes max for one (batch, feature) combination
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    
    # Only compute for valid batch and feature
    if batch_id >= n_batch or feature_id >= n_features:
        return
    
    # Calculate base offset for this batch and feature
    base_offset = batch_id * n_features * n_positions + feature_id * n_positions
    
    # Find max along positions dimension
    max_val = -float('inf')
    for pos in range(n_positions):
        current_offset = base_offset + pos
        current_val = tl.load(input_ptr + current_offset)
        max_val = tl.maximum(max_val, current_val)
    
    # Store the max value at position 0 in the output
    output_offset = batch_id * n_features * 1 + feature_id * 1
    tl.store(output_ptr + output_offset, max_val)

@torch.fx.wrap
def simple_max_kernel_wrapper(in_0):
    # Get input shape [batch, features, positions]
    batch_size, n_features, n_positions = in_0.shape
    
    # Create output tensor with reduced dimension [batch, features, 1]
    out = torch.empty((batch_size, n_features, 1), dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid: one program per batch and feature combination
    grid_z = batch_size
    grid_y = n_features
    grid_x = 1  # No need for programs along positions dimension since we're reducing it
    
    # Launch kernel - block size doesn't matter since we're not using it
    max_along_dim_kernel[(grid_z, grid_y)](
        input_ptr=in_0,
        output_ptr=out,
        n_batch=batch_size,
        n_features=n_features,
        n_positions=n_positions,
        BLOCK_SIZE=1,  # Not actually used in this kernel
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return simple_max_kernel_wrapper