import torch
import triton
import triton.language as tl

# Pattern matching function - match the sum operation
def pattern(in_1):
    tmp_0 = in_1.sum(dim = 2, keepdim = True)
    return tmp_0

# Argument extraction function
def replacement_args(in_1):
    return (in_1,)

# Optimized kernel for sum along dimension 2
@triton.jit
def sum_dim2_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes one (batch, channel, height) combination
    pid = tl.program_id(0)
    
    # Calculate indices for batch, channel, and height
    batch_idx = pid // (n_channels * height)
    channel_idx = (pid // height) % n_channels  
    height_idx = pid % height
    
    # Compute input offset for this (batch, channel, height)
    input_offset = (batch_idx * n_channels * height * width + 
                   channel_idx * height * width + 
                   height_idx * width)
    
    # Load elements along dimension 2 (width) for this position
    offsets = input_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (input_offset + width)
    
    # Load and sum the width dimension
    input_slice = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    sum_val = tl.sum(input_slice)
    
    # Store the sum at the corresponding position in output tensor
    # Output shape is [batch, channels, height, 1]
    output_offset = (batch_idx * n_channels * height + 
                    channel_idx * height + 
                    height_idx)
    tl.store(output_ptr + output_offset, sum_val)

@torch.fx.wrap
def sum_dim2_optimized(in_1):
    """Optimized sum along dimension 2"""
    batch, channels, height, width = in_1.shape
    
    # Create output tensor with shape [batch, channels, height, 1]
    output = torch.empty((batch, channels, height, 1), dtype=in_1.dtype, device=in_1.device)
    
    # Use very small block size to match tiny tensor size
    BLOCK_SIZE = min(width, 32)  # Use 32 or smaller for small tensors
    
    # Total number of (batch, channel, height) combinations  
    total_elements = batch * channels * height
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch optimized kernel
    if num_programs > 0:
        sum_dim2_kernel[(num_programs,)](
            input_ptr=in_1,
            output_ptr=output,
            n_batch=batch,
            n_channels=channels,
            height=height,
            width=width,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output

# Replacement function
def replacement_func():
    return sum_dim2_optimized