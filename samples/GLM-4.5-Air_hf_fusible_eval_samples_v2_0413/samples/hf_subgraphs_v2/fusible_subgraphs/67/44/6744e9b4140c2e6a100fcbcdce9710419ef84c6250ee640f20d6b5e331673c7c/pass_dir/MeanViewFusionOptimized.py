import torch
import triton
import triton.language as tl

# Pattern matching function for mean + view fusion
def pattern(x):
    """Match the pattern: mean over spatial dims followed by view(1, 1, -1)"""
    tmp1 = x.mean((2, 3))
    tmp2 = tmp1.view(1, 1, -1)
    return tmp1, tmp2

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized Triton kernel for mean + view fusion
@triton.jit
def mean_view_fused_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    num_channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    """Kernel that computes mean and reshapes to (1, 1, -1) in one operation"""
    pid = tl.program_id(0)
    
    # Each program handles one channel output (since output is [1, 1, num_channels])
    if pid >= num_channels:
        return
    
    # Calculate mean for this channel across all batches and spatial dimensions
    total_elements = batch_size * height * width
    channel_sum = 0.0
    
    # Sum across all batch elements and spatial locations for this channel
    for b in range(batch_size):
        for h in range(height):
            for w in range(width):
                ptr = input_ptr + (b * num_channels * height * width + 
                                  pid * height * width + h * width + w)
                val = tl.load(ptr)
                channel_sum += val
    
    # Compute mean
    channel_mean = channel_sum / total_elements
    
    # Store result directly in the final output shape [1, 1, num_channels]
    # Since output is (1, 1, num_channels), we just store at position pid
    tl.store(output_ptr + pid, channel_mean)

@torch.fx.wrap
def fused_mean_view_optimized(x):
    """
    Fused mean + view operation with optimized Triton kernel
    Input: [batch_size, num_channels, height, width] 
    Output: [1, 1, num_channels] - mean over spatial and batch dimensions
    """
    batch_size, num_channels, height, width = x.shape
    
    # Output is [1, 1, num_channels] - flatten to [num_channels] for efficient storage
    output = torch.empty(num_channels, dtype=x.dtype, device=x.device)
    
    # Use optimal block size for better GPU utilization
    BLOCK_SIZE = 256
    grid = (num_channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    mean_view_fused_kernel[grid](
        input_ptr=x,
        output_ptr=output,
        batch_size=batch_size,
        num_channels=num_channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Reshape to final output shape [1, 1, num_channels]
    return output.view(1, 1, -1)

# Replacement function (must return a callable function)
def replacement_func():
    return fused_mean_view_optimized