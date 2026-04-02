import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern matching: max -> extract -> expand -> subtract sequence for softmax
    This mirrors the computation from model.py:
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]  
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    """
    tmp_0 = torch.max(in_0, -1, keepdim=True)
    tmp_1 = tmp_0[0]
    tmp_2 = tmp_1.expand_as(in_0)
    tmp_3 = tmp_2 - in_0
    return tmp_3

def replacement_args(in_0):
    """Extract arguments for replacement function"""
    return (in_0,)

@triton.jit
def fused_max_subtract_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel that computes max and then x - max in a single pass
    This reduces memory bandwidth by avoiding intermediate storage
    """
    # Compute global program ID
    pid = tl.program_id(0)
    
    # Calculate total elements in the tensor
    total_elements = batch_size * channels * spatial_size
    
    # Check bounds
    if pid >= total_elements:
        return
    
    # Convert 1D program ID to batch, channel, and spatial coordinates
    spatial_idx = pid % spatial_size
    channel_idx = (pid // spatial_size) % channels
    batch_idx = pid // (channels * spatial_size)
    
    # Calculate global offset for this element
    global_offset = pid
    
    # Load the spatial dimension for this batch-channel pair
    spatial_offsets = tl.arange(0, BLOCK_SIZE)
    base_offset = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx
    element_offsets = base_offset + spatial_offsets
    
    # Load spatial chunk with masking
    mask = element_offsets < total_elements
    x = tl.load(x_ptr + element_offsets, mask=mask, other=-float('inf'))
    
    # Compute max over the spatial dimension
    max_val = tl.max(x)
    
    # Compute x - max for numerical stability
    result = x - max_val
    
    # Store results
    tl.store(out_ptr + element_offsets, result, mask=mask)

@torch.fx.wrap
def fused_max_subtract_func(x):
    """Function that applies the fused max-subtract operation"""
    # Get tensor shape
    batch_size, channels, spatial_size = x.shape
    
    # Use fixed compile-time block size that works for all our cases
    # Choose a block size that's a power of 2 for good GPU efficiency
    BLOCK_SIZE = 256  # Fixed block size
    
    # Calculate total elements and number of programs needed
    total_elements = batch_size * channels * spatial_size
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Launch kernel with 1D grid
    fused_max_subtract_kernel[(num_programs,)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        spatial_size=spatial_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return fused_max_subtract_func