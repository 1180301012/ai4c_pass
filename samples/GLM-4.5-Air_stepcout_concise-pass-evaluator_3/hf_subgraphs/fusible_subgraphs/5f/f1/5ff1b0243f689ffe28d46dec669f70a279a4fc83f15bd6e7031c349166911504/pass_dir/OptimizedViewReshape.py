import torch
import triton
import triton.language as tl

# Pattern matching function - matches the view operation for 4D to 3D reshape
def pattern(in_1):
    tmp_5 = in_1.view(1, 512, -1)
    return tmp_5

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

@triton.jit
def optimized_view_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    n_features,
    original_h,
    original_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one element in the flattened spatial dimensions
    batch_id = tl.program_id(0)
    feature_id = tl.program_id(1)
    linear_id = tl.program_id(2)
    
    # Calculate output dimensions
    spatial_size = original_h * original_w
    if linear_id >= spatial_size:
        return
    
    # Calculate offsets
    batch_offset = batch_id * n_features * spatial_size
    feature_offset = feature_id * spatial_size
    spatial_offset = linear_id
    
    output_offset = batch_offset + feature_offset + spatial_offset
    
    # Calculate input offset (4D -> 3D mapping)
    input_spatial_id = linear_id
    input_h = input_spatial_id // original_w
    input_w = input_spatial_id % original_w
    
    input_batch_offset = batch_id * n_features * original_h * original_w
    input_feature_offset = feature_id * original_h * original_w
    input_spatial_offset = input_h * original_w + input_w
    
    input_offset = input_batch_offset + input_feature_offset + input_spatial_offset
    
    # Load input value and store to output
    val = tl.load(input_ptr + input_offset)
    tl.store(output_ptr + output_offset, val)

@torch.fx.wrap
def optimized_view_reshape(in_1):
    # Get input shape
    batch_size, n_features, h, w = in_1.shape
    spatial_size = h * w
    
    # Create output tensor with optimized memory layout
    out = torch.empty((batch_size, n_features, spatial_size), dtype=in_1.dtype, device=in_1.device)
    
    # Grid configuration
    BLOCK_SIZE = 256
    grid_z = batch_size
    grid_y = n_features
    grid_x = (spatial_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    optimized_view_kernel[(grid_z, grid_y, grid_x)](
        input_ptr=in_1,
        output_ptr=out,
        batch_size=batch_size,
        n_features=n_features,
        original_h=h,
        original_w=w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return optimized_view_reshape