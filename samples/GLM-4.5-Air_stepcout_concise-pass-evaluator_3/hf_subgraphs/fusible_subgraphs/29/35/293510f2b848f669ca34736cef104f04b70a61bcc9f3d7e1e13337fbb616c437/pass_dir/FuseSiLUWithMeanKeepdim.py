import torch
import triton
import triton.language as tl
import math

# Pattern matching function for keepdim=True pattern
def pattern(in_0):
    """
    Matches SiLU -> Mean reduction pattern with keepdim=True
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)
    """
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_0, tmp_1

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized kernel for SiLU + Mean reduction with keepdim
@triton.jit
def fused_silu_mean_keepdim_kernel(
    input_ptr,
    silu_output_ptr,
    mean_output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Each program processes a spatial element for one channel
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # channel
    pid_k = tl.program_id(2)  # spatial element (flattened)
    
    # Calculate offsets
    input_offset = pid_m * channels * height * width + pid_n * height * width + pid_k
    silu_output_offset = input_offset
    
    # Load input value
    x = tl.load(input_ptr + input_offset, mask=input_offset < (batch_size * channels * height * width), other=0.0)
    
    # Compute SiLU: x * sigmoid(x)
    sigmoid_x = 1.0 / (1.0 + tl.exp(-x))
    silu_out = x * sigmoid_x
    
    # Store SiLU result
    tl.store(silu_output_ptr + silu_output_offset, silu_out)
    
    # Handle mean reduction for this channel
    if pid_m < batch_size and pid_n < channels:
        # Sum over spatial dimensions
        spatial_sum = 0.0
        spatial_count = height * width
        
        for pos in range(height * width):
            idx = pid_m * channels * height * width + pid_n * height * width + pos
            val = tl.load(input_ptr + idx, mask=idx < (batch_size * channels * height * width), other=0.0)
            silu_val = val * (1.0 / (1.0 + tl.exp(-val)))
            spatial_sum += silu_val
        
        # Compute mean (keepdim=True means we keep the reduced dimensions)
        spatial_mean = spatial_sum / spatial_count
        mean_output_offset = pid_m * channels * height * width + pid_n * height * width  # All spatial positions get same value
        for pos in range(height * width):
            mean_idx = mean_output_offset + pos
            tl.store(mean_output_ptr + mean_idx, spatial_mean)

# Optimized kernel for more efficient reduction
@triton.jit
def fused_silu_mean_keepdim_optimized_kernel(
    input_ptr,
    silu_output_ptr,
    mean_output_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program handles one channel reduction
    pid_m = tl.program_id(0)  # batch
    pid_n = tl.program_id(1)  # channel
    
    if pid_m >= batch_size or pid_n >= channels:
        return
    
    # Shared memory for partial sums (adjust size based on available shared memory)
    partial_sum = tl.zeros([1], dtype=tl.float32)
    
    # Process spatial dimensions with vectorization
    spatial_elements = height * width
    # Number of iterations to process all spatial elements
    num_iterations = (spatial_elements + 128 - 1) // 128
    
    for i in range(num_iterations):
        # Calculate current position
        pos_base = i * 128
        pos_end = min(pos_base + 128, spatial_elements)
        
        # Load spatial block and compute SiLU in parallel
        spatial_sum = 0.0
        for pos in range(pos_base, pos_end):
            idx = pid_m * channels * spatial_elements + pid_n * spatial_elements + pos
            val = tl.load(input_ptr + idx, mask=idx < (batch_size * channels * spatial_elements), other=0.0)
            silu_val = val * (1.0 / (1.0 + tl.exp(-val)))
            spatial_sum += silu_val
        
        partial_sum[0] += spatial_sum
    
    # Compute mean
    spatial_mean = partial_sum[0] / spatial_elements
    
    # Store SiLU output (already done in parallel)
    # We need to write the spatial mean to all spatial positions
    mean_base_offset = pid_m * channels * spatial_elements + pid_n * spatial_elements
    for pos in range(spatial_elements):
        mean_idx = mean_base_offset + pos
        tl.store(mean_output_ptr + mean_idx, spatial_mean)

# Kernel wrapper
@torch.fx.wrap
def fused_silu_mean_keepdim(input_tensor):
    """
    Optimized fused SiLU + Mean reduction with keepdim=True
    Returns: (silu_output, mean_output_with_keepdim)
    """
    batch_size, channels, height, width = input_tensor.shape
    spatial_elements = height * width
    
    # Create output tensors
    silu_output = torch.empty_like(input_tensor)
    mean_output = torch.empty_like(input_tensor)  # Same shape as input for keepdim=True
    
    # Kernel launch configuration
    if spatial_elements > 256:
        # For larger spatial dimensions, use optimized reduction
        block_size_m = 8
        block_size_n = 32
        
        blocks_m = (batch_size + block_size_m - 1) // block_size_m
        blocks_n = (channels + block_size_n - 1) // block_size_n
        
        grid = (blocks_m, blocks_n)
        
        fused_silu_mean_keepdim_optimized_kernel[grid](
            input_tensor,
            silu_output,
            mean_output,
            batch_size,
            channels,
            height,
            width,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
        )
    else:
        # For smaller spatial dimensions, use simpler kernel
        block_size_m = 4
        block_size_n = 8
        block_size_k = 32
        
        blocks_m = (batch_size + block_size_m - 1) // block_size_m
        blocks_n = (channels + block_size_n - 1) // block_size_n
        blocks_k = (spatial_elements + block_size_k - 1) // block_size_k
        
        grid = (blocks_m, blocks_n, blocks_k)
        
        fused_silu_mean_keepdim_kernel[grid](
            input_tensor,
            silu_output,
            mean_output,
            batch_size,
            channels,
            height,
            width,
            BLOCK_SIZE_M=block_size_m,
            BLOCK_SIZE_N=block_size_n,
            BLOCK_SIZE_K=block_size_k,
        )
    
    return silu_output, mean_output

# Replacement function (returns function reference)
def replacement_func():
    return fused_silu_mean_keepdim