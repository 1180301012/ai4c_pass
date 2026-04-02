import torch
import triton
import triton.language as tl
import math

def pattern(in_0):
    """
    Pattern matching: softmax operation for the final softmax step
    This mirrors the computation from model.py:
    tmp_4 = torch.nn.functional.softmax(tmp_3, dim=-1)
    """
    tmp_4 = torch.nn.functional.softmax(in_0, dim=-1)
    return tmp_4

def replacement_args(in_0):
    """Extract arguments for replacement function"""
    return (in_0,)

@triton.jit
def optimized_softmax_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized softmax kernel that computes exp(x) / sum(exp(x)) efficiently
    Uses numerical stability and optimized memory access patterns
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
    
    # Calculate base offset for this element in its spatial dimension
    base_offset = (batch_idx * channels + channel_idx) * spatial_size + spatial_idx
    
    # Load spatial chunk with masking
    spatial_offsets = tl.arange(0, BLOCK_SIZE)
    element_offsets = base_offset + spatial_offsets
    mask = element_offsets < total_elements
    x = tl.load(x_ptr + element_offsets, mask=mask, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(x)
    
    # Compute exp(x - max)
    exp_x = tl.exp(x - max_val)
    
    # Compute sum of exp(x - max)
    sum_exp = tl.sum(exp_x)
    
    # Compute softmax: exp(x - max) / sum(exp(x - max))
    softmax_result = exp_x / sum_exp
    
    # Store results
    tl.store(out_ptr + element_offsets, softmax_result, mask=mask)

@torch.fx.wrap
def optimized_softmax_func(x):
    """Function that applies the optimized softmax operation"""
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
    optimized_softmax_kernel[(num_programs,)](
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
    return optimized_softmax_func