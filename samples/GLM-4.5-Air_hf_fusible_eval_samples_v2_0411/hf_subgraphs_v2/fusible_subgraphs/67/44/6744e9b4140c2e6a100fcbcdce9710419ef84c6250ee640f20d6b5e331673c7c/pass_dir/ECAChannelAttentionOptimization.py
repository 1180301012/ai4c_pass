import torch
import triton
import triton.language as tl

# Pattern matching function - simplest possible test pattern
def pattern(x):
    # Match a simple operation - mean
    result = x.mean((2, 3))
    return result

# Argument extraction function
def replacement_args(x):
    return (x,)

# Triton kernel for mean reduction
@triton.jit
def mean_kernel(
    input_ptr, output_ptr,
    batch, channels, height, width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Compute which channel this program handles
    c = pid
    if c >= channels:
        return
    
    # Initialize spatial sum for this channel across all batches
    spatial_sum = 0.0
    
    # Compute spatial mean for all batches
    for b in range(batch):
        # Calculate base offset for this batch and channel
        batch_offset = b * channels * height * width
        channel_offset = batch_offset + c * height * width
        
        # Sum over spatial dimensions
        for h in range(height):
            row_offset = channel_offset + h * width
            for w in range(width):
                val = tl.load(input_ptr + row_offset + w)
                spatial_sum += val
    
    # Compute mean: sum / (height * width)
    spatial_mean = spatial_sum / (height * width)
    
    # Store result (for single batch channel)
    tl.store(output_ptr + c, spatial_mean)

# Proper replacement function using Triton kernel
@torch.fx.wrap
def optimized_mean_optimization(x):
    """Optimized mean computation using Triton"""
    
    # Ensure tensor is on GPU and contiguous
    if x.device.type != 'cuda':
        x = x.cuda()
    x = x.contiguous()
    
    # Get input shape
    if len(x.shape) == 4:
        batch, channels, height, width = x.shape
    else:
        raise ValueError("Input must be 4D tensor (B, C, H, W)")
    
    # Allocate output on same device using allowed APIs
    mean_output = torch.empty(channels, dtype=torch.float32, device=x.device)
    mean_output = mean_output.contiguous()
    
    BLOCK_SIZE = 1024
    
    # Launch Triton kernel
    n_programs = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    mean_kernel[(n_programs,)](
        input_ptr=x,
        output_ptr=mean_output,
        batch=batch,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return mean_output

# Replacement function
def replacement_func():
    return optimized_mean_optimization