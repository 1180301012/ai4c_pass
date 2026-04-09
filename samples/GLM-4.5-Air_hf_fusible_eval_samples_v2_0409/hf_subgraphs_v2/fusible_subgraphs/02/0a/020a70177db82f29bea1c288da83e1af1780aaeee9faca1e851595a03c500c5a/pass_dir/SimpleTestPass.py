import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    """Simple test pattern"""
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def simple_test_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple test kernel for adaptive_avg_pool2d with size 1"""
    pid = tl.program_id(0)
    
    # Each program handles one batch element
    batch_offset = pid * channels
    
    # Process channels in blocks
    channel_idx = tl.program_id(1)
    
    # Check if this channel is within bounds
    if channel_idx >= channels:
        return
    
    # Compute base output index for this channel
    output_idx = batch_offset + channel_idx
    
    # Sum all spatial locations for this channel
    spatial_sum = 0.0
    
    # Iterate through all spatial positions
    for h_idx in range(height):
        for w_idx in range(width):
            # Linear index for this spatial position and channel
            input_idx = batch_offset + (h_idx * width + w_idx) * channels + channel_idx
            
            # Load input value (no mask needed since we checked bounds)  
            input_val = tl.load(input_ptr + input_idx)
            
            # Accumulate sum
            spatial_sum += input_val
    
    # Compute average (global average pooling)
    element_count = height * width
    avg_val = spatial_sum / element_count if element_count > 0 else 0.0
    
    # Store result (always store since we checked channel bounds)
    tl.store(output_ptr + output_idx, avg_val)

@torch.fx.wrap
def simple_test_replacement(input_tensor):
    """Simple replacement function using Triton"""
    batch_size, channels, height, width = input_tensor.shape
    
    # Use 2D grid: each kernel program handles one batch element one channel
    grid = (batch_size, channels, 1)
    
    # Create output tensor
    output = torch.empty((batch_size, channels), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    simple_test_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=128,  # Not used in this kernel version
    )
    
    return output

def replacement_func():
    return simple_test_replacement