import torch
import triton
import triton.language as tl

def pattern(in_3):
    tmp_5 = in_3.sum(dim=3, keepdim=True)
    tmp_6 = in_3 / tmp_5
    return tmp_6

def replacement_args(in_3):
    return (in_3,)

@triton.jit
def fused_sum_div_kernel(
    input_ptr, out_ptr,
    batch_size, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel, height) combination
    pid = tl.program_id(0)
    
    # Extract batch, channel, height from program ID  
    bch_id = pid
    batch = bch_id // (channels * height)
    remainder = bch_id % (channels * height)
    channel = remainder // height
    h = remainder % height
    
    # Load the entire width slice for this (batch, channel, height)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < width
    
    input_slice = tl.load(input_ptr + 
                         batch * channels * height * width + 
                         channel * height * width + 
                         h * width + offsets, 
                         mask=mask, other=0.0)
    
    # Compute sum of this width slice
    sum_val = tl.sum(input_slice)
    
    # Add epsilon for numerical stability and normalize each element
    normalized_slice = input_slice / (sum_val + 1e-8)
    
    # Store result in same location
    tl.store(out_ptr + 
             batch * channels * height * width + 
             channel * height * width + 
             h * width + offsets, 
             normalized_slice, mask=mask)

@torch.fx.wrap  
def fused_sum_div(in_3):
    batch_size, channels, height, width = in_3.shape
    
    # One kernel per (batch, channel, height) combination
    num_blocks = batch_size * channels * height
    
    # Each block processes a slice of width elements
    BLOCK_SIZE = min(1024, width)
    grid_size = (num_blocks,)
    
    # Create output tensor
    output = torch.empty_like(in_3)
    
    # Launch kernel - one block per (batch, channel, height) combination
    fused_sum_div_kernel[grid_size](
        in_3,
        output,
        batch_size, channels, height, width,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_sum_div