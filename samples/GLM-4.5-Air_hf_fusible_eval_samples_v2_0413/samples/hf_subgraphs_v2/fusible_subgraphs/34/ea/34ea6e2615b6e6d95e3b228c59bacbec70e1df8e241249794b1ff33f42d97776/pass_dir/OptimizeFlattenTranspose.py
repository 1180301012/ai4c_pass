import torch
import triton
import triton.language as tl

@triton.jit
def flatten_transpose_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    in_channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Get program IDs
    batch_idx = tl.program_id(0)
    feature_idx = tl.program_id(1)
    
    # Calculate total elements per batch flattened dimension
    flattened_dim = height * width
    batch_offset = batch_idx * flattened_dim * in_channels
    
    # Calculate output position 
    output_offset = batch_idx * flattened_dim * in_channels + feature_idx * flattened_dim
    
    # Process a block of data
    x_start = feature_idx * BLOCK_SIZE_X
    y_start = 0
    
    if x_start >= in_channels:
        return
        
    for x in range(x_start, min(x_start + BLOCK_SIZE_X, in_channels)):
        for y in range(y_start, min(y_start + BLOCK_SIZE_Y, flattened_dim)):
            # Calculate input position: [batch, channel, h, w]
            input_idx = batch_offset + x * flattened_dim + y
            
            # Calculate output position: [batch, flattened_dim, channel] -> [batch, h*w, channel]
            output_idx = output_offset + y * in_channels + x
            
            # Load input and store to output
            val = tl.load(input_ptr + input_idx)
            tl.store(output_ptr + output_idx, val)

@torch.fx.wrap
def optimized_flatten_transpose(input_tensor):
    # Input shape: [batch, channels, height, width]
    # Output shape: [batch, height*width, channels]
    batch_size, in_channels, height, width = input_tensor.shape
    flattened_dim = height * width
    
    # Create output tensor
    output_shape = [batch_size, flattened_dim, in_channels]
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    BLOCK_SIZE_X = 32  # Process multiple channels
    BLOCK_SIZE_Y = 32  # Process multiple spatial positions
    
    grid_x = triton.cdiv(in_channels, BLOCK_SIZE_X)
    grid_y = triton.cdiv(flattened_dim, BLOCK_SIZE_Y)
    grid = (batch_size, grid_x * grid_y)  # Each batch element processed separately
    
    # Launch kernel
    flatten_transpose_kernel[grid](
        input_tensor,
        output,
        batch_size, in_channels, height, width,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
    )
    
    return output

# Pattern matching function - matches flatten + transpose sequence
def pattern(input_tensor):
    tmp_6 = input_tensor.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    def optimized_forward(input_tensor):
        # Use optimized flatten + transpose
        tmp_7 = optimized_flatten_transpose(input_tensor)
        return tmp_7
    
    return optimized_forward