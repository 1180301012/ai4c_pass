import torch
import triton
import triton.language as tl
import math

def pattern(in_5, start_channel):
    """
    Pattern for tensor slicing operation: extracting channels from start_channel to end
    Matches: tmp_4 = in_5[slice(None, None, None), slice(C, None, None), slice(None, None, None), slice(None, None, None)]
    """
    return in_5[slice(None, None, None), slice(start_channel, None, None), slice(None, None, None), slice(None, None, None)]

def replacement_args(in_5, start_channel):
    return (in_5, start_channel)

@triton.jit
def optimized_slice_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    input_channels, 
    height,
    width,
    start_channel,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    """
    Optimized Triton kernel for tensor channel slicing
    Each program processes a block of spatial dimensions (height, width)
    """
    # Initialize program IDs for spatial dimensions
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_b = tl.program_id(2)  # batch dimension
    
    # Calculate spatial offsets within this block
    y_offset = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    x_offset = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    # Create coordinate offsets
    y_coords = y_offset[:, None]
    x_coords = x_offset[None, :]
    
    # Calculate mask for valid spatial coordinates
    y_mask = y_coords < height
    x_mask = x_coords < width
    spatial_mask = y_mask & x_mask
    
    # Number of output channels (after slicing)
    output_channels = input_channels - start_channel
    
    # Process each channel in the output
    for c in range(0, output_channels, 1):  # Can be optimized with tiling if needed
        # Calculate input channel index
        input_c = start_channel + c
        
        # Calculate memory addresses
        input_base = pid_b * input_channels * height * width + input_c * height * width
        output_base = pid_b * output_channels * height * width + c * height * width
        
        # Read input data
        input_addr = input_ptr + input_base + y_coords * width + x_coords
        output_addr = output_ptr + output_base + y_coords * width + x_coords
        
        # Load with spatial mask
        data = tl.load(input_addr, mask=spatial_mask, other=0.0)
        
        # Store to output with spatial mask
        tl.store(output_addr, data, mask=spatial_mask)

@torch.fx.wrap
def optimized_tensor_slice(input_tensor, start_channel):
    """
    Optimized tensor slice implementation using Triton
    """
    batch_size, input_channels, height, width = input_tensor.shape
    output_channels = input_channels - start_channel
    
    # Create output tensor
    output = torch.empty((batch_size, output_channels, height, width), 
                        dtype=input_tensor.dtype, 
                        device=input_tensor.device)
    
    # Set optimal block sizes based on tensor dimensions
    BLOCK_SIZE_Y = 32
    BLOCK_SIZE_X = 32
    
    # Calculate grid dimensions
    grid_y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_b = batch_size
    
    # Launch kernel
    optimized_slice_kernel[(grid_y, grid_x, grid_b)](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        height=height,
        width=width,
        start_channel=start_channel,
        BLOCK_SIZE_Y=BLOCK_SIZE_Y,
        BLOCK_SIZE_X=BLOCK_SIZE_X,
    )
    
    return output

def replacement_func():
    return optimized_tensor_slice