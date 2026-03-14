import torch
import triton
import triton.language as tl

def pattern(spatial_input):
    """
    Pattern matching: Flatten + Transpose
    This handles the spatial feature processing pipeline
    """
    tmp_6 = spatial_input.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    return tmp_7

def replacement_args(spatial_input):
    """Extract arguments needed for the replacement function"""
    return (spatial_input,)

@triton.jit
def optimized_flatten_transpose_kernel(
    input_ptr, output_ptr,
    batch_size, channels, height, width, total_hw,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized Triton kernel for flatten + transpose operations
    Input: (batch, channels, height, width)
    Output: (batch, height*width, channels)
    """
    pid = tl.program_id(0)
    
    if pid >= batch_size * total_hw:
        return
    
    # Calculate batch index and flattened spatial index
    batch_idx = pid // total_hw
    spatial_idx = pid % total_hw
    
    # Input offset: (batch, channel, h, w) -> batch * C * H * W + c * H * W + h * W + w
    # Output offset: (batch, h*w, channel) -> batch * H * W * C + (h * W + w) * C + c
    
    # Process all channels for this spatial position
    for c in range(channels):
        # Input offset calculation
        input_offset = (
            batch_idx * channels * height * width +
            c * height * width +
            (spatial_idx // width) * width +  # h * width + w
            (spatial_idx % width)             # w
        )
        
        # Output offset calculation  
        output_offset = (
            batch_idx * total_hw * channels +
            spatial_idx * channels +
            c
        )
        
        # Load input value
        val = tl.load(input_ptr + input_offset, mask=input_offset < (batch_size * channels * height * width), other=0.0)
        
        # Store to transposed position
        tl.store(output_ptr + output_offset, val, mask=output_offset < (batch_size * total_hw * channels))

@torch.fx.wrap  
def optimized_flatten_transpose(spatial_input):
    """
    Optimized flatten + transpose implementation using Triton
    """
    batch_size = spatial_input.shape[0]
    channels = spatial_input.shape[1] 
    height = spatial_input.shape[2]
    width = spatial_input.shape[3]
    total_hw = height * width
    
    # Output shape: (batch, height*width, channels)
    output_shape = (batch_size, total_hw, channels)
    output = torch.empty(output_shape, dtype=torch.float32, device=spatial_input.device)
    
    # Configure block size for good GPU occupancy
    BLOCK_SIZE = 256  # Good balance between shared memory and parallelism
    
    # Calculate grid size
    total_elements = batch_size * total_hw
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized kernel - wrap grid_size in a tuple
    optimized_flatten_transpose_kernel[(grid_size,)](
        input_ptr=spatial_input,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        total_hw=total_hw,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    """Return the optimized function implementation"""
    return optimized_flatten_transpose