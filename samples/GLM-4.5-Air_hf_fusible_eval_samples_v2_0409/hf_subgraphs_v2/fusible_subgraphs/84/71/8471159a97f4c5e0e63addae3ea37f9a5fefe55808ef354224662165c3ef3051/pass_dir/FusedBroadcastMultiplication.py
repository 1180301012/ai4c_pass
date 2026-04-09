import torch
import triton
import triton.language as tl
import math

@triton.jit
def fused_broadcast_mult_kernel(
    broadcast_ptr, input_ptr, output_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate linear index for output
    linear_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    if linear_idx >= batch_size * channels * height * width:
        return
    
    # Convert linear index to NCHW coordinates
    w_idx = linear_idx % width
    h_idx = (linear_idx // width) % height
    c_idx = (linear_idx // (width * height)) % channels
    b_idx = linear_idx // (width * height * channels)
    
    # Load broadcast value (same for all H, W positions for each channel)
    # The broadcast tensor has shape [batch_size, channels, 1, 1]
    broadcast_offset = b_idx * channels + c_idx
    broadcast_value = tl.load(broadcast_ptr + broadcast_offset)
    
    # Load input value  
    input_offset = linear_idx
    input_value = tl.load(input_ptr + input_offset)
    
    # Perform multiplication
    output_value = input_value * broadcast_value
    
    # Store output
    tl.store(output_ptr + linear_idx, output_value)

@torch.fx.wrap
def fused_broadcast_multiply(broadcast_tensor, input_tensor):
    """
    Optimized fused broadcasting multiplication
    broadcast_tensor: [batch_size, channels, 1, 1]
    input_tensor: [batch_size, channels, height, width]
    """
    batch_size, channels, height, width = input_tensor.shape
    
    # Use optimal block size for maximum GPU occupancy
    BLOCK_SIZE = 1024  # Adjust based on typical GPU architecture
    grid_size = (batch_size * channels * height * width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output
    output = torch.empty_like(input_tensor)
    
    # Launch kernel
    fused_broadcast_mult_kernel[grid_size](
        broadcast_ptr=broadcast_tensor.reshape(-1),  # Flatten to [batch_size * channels]
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def pattern(broadcast_tensor, input_tensor):
    return input_tensor * broadcast_tensor

def replacement_args(broadcast_tensor, input_tensor):
    return (broadcast_tensor, input_tensor)

def replacement_func():
    return fused_broadcast_multiply