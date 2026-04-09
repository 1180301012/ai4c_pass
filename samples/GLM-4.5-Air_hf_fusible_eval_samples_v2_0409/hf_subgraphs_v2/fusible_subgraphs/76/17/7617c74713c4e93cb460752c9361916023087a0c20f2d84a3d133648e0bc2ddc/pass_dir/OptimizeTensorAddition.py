import torch
import triton
import triton.language as tl

def pattern(input_5, input_4):
    """Pattern to match tensor addition operation"""
    return input_5 + input_4

def replacement_args(input_5, input_4):
    """Extract arguments for addition operation"""
    return (input_5, input_4)

@triton.jit
def optimized_add_kernel(
    input5_ptr,
    input4_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized addition kernel that processes multiple elements efficiently"""
    pid = tl.program_id(0)
    
    if pid >= batch_size:
        return
    
    # Process all channels and spatial dimensions for this batch element
    for c in range(channels):
        for h in range(height):
            for w in range(width):
                # Calculate offsets for input tensors
                input5_offset = (pid * channels * height * width) + (c * height * width) + (h * width) + w
                input4_offset = input5_offset  # Same shape as input5
                
                # Load values from both tensors
                input5_val = tl.load(input5_ptr + input5_offset)
                input4_val = tl.load(input4_ptr + input4_offset)
                
                # Add and store result
                result = input5_val + input4_val
                tl.store(out_ptr + input5_offset, result)

@torch.fx.wrap
def optimized_add(input_5, input_4):
    """Optimized addition function"""
    batch_size, channels, height, width = input_4.shape
    grid_size = batch_size
    
    # Create output tensor
    out = torch.empty_like(input_4)
    
    # Launch kernel with corrected parameter order
    optimized_add_kernel[grid_size](
        input5_ptr=input_5,
        input4_ptr=input_4,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=1
    )
    
    return out

def replacement_func():
    """Return the optimized addition function"""
    return optimized_add