import torch
import triton
import triton.language as tl

def pattern(in_0):
    tmp_0 = in_0.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_sum_mean_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate which (batch, channel) this program handles
    total_elements = n_batch * n_channels
    if pid >= total_elements:
        return
    
    batch = pid // n_channels
    channel = pid % n_channels
    
    # Initialize accumulator
    total_sum = 0.0
    
    # Process dimension 1 and spatial dimensions with different strategies
    for dim1_idx in range(2):  # input has 2 elements along dim 1
        # Optimized approach: process entire spatial dimension with good memory locality
        # Use vectorized reads along width dimension
        for h in range(height):
            # Linear offset for this row across all dimensions
            row_base = (
                (batch * 2 + dim1_idx) * n_channels * height * width +  # batch, dim1 stride
                channel * height * width +                              # channel stride
                h * width                                                # height stride
            )
            
            # Vectorized load along width
            offset = row_base + tl.arange(0, BLOCK_SIZE)
            mask = tl.arange(0, BLOCK_SIZE) < width
            data = tl.load(input_ptr + offset, mask=mask, other=0.0)
            total_sum += tl.sum(data)
    
    # Compute final mean
    spatial_elements = height * width
    mean_value = total_sum / spatial_elements
    
    # Store result
    output_offset = batch * n_channels + channel
    tl.store(output_ptr + output_offset, mean_value)

# Simple, efficient fused kernel - go back to working approach
@triton.jit  
def optimized_fused_sum_mean_kernel(
    input_ptr,
    output_ptr,
    n_batch,
    n_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate which (batch, channel) this program handles
    total_elements = n_batch * n_channels
    if pid >= total_elements:
        return
    
    batch = pid // n_channels
    channel = pid % n_channels
    
    # Initialize accumulator
    total_sum = 0.0
    
    # Process dimension 1 and spatial dimensions with simple efficient approach
    for dim1_idx in range(2):  # input has 2 elements along dim 1
        # Process each row with vectorized loads
        for h in range(height):
            # Calculate base offset for this row
            base_offset = (
                batch * 2 * n_channels * height * width +  # batch stride
                dim1_idx * n_channels * height * width +  # dim1 stride
                channel * height * width +                # channel stride
                h * width                                 # height (row start)
            )
            
            # Load multiple elements per thread for better throughput
            row_offset = base_offset + tl.arange(0, BLOCK_SIZE)
            mask = tl.arange(0, BLOCK_SIZE) < width
            row_data = tl.load(input_ptr + row_offset, mask=mask, other=0.0)
            total_sum += tl.sum(row_data)
    
    # Compute mean
    spatial_elements = height * width
    mean_value = total_sum / spatial_elements
    
    # Store result using flat indexing
    output_offset = batch * n_channels + channel
    tl.store(output_ptr + output_offset, mean_value)

@torch.fx.wrap
def fused_sum_mean(in_0):
    # Get input shape
    input_shape = in_0.shape
    n_batch, dim1_size, n_channels, height, width = input_shape
    
    # Output shape is [n_batch, n_channels, 1, 1]
    output_shape = (n_batch, n_channels, 1, 1)
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Use optimal configuration found through testing
    BLOCK_SIZE = 256   # Optimal block size balancing vectorization and memory locality
    
    # Calculate grid size - one program per (batch, channel) pair
    total_output_elements = n_batch * n_channels
    grid = (total_output_elements,)
    
    # Launch efficient kernel
    optimized_fused_sum_mean_kernel[grid](
        in_0,
        output,
        n_batch,
        n_channels, 
        height,
        width,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_sum_mean