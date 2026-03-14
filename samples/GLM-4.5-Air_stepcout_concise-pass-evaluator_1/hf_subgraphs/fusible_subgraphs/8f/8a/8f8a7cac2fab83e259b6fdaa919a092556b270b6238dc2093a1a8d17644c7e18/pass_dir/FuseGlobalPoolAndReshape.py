import torch
import triton
import triton.language as tl

# Pattern matching function - simple test to see if adaptive_avg_pool2d matches
def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

# Optimized Triton kernel that fuses global average pooling and flatten operations
@triton.jit
def fused_global_pool_flatten_kernel(
    x_ptr,
    out_ptr,
    n_batch,
    n_channels,
    height,
    width,
    spatial_size: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Each program handles one batch * channel combination
    batch_idx = pid // n_channels
    channel_idx = pid % n_channels
    
    if batch_idx >= n_batch or channel_idx >= n_channels:
        return
    
    # Compute the starting position of this batch-channel combination in the input
    input_offset = ( batch_idx * n_channels + channel_idx ) * spatial_size
    
    # Compute the mean across spatial dimensions using a loop
    sum_val = 0.0
    for i in range(0, spatial_size, BLOCK_SIZE):
        # Load a block of elements
        offsets = input_offset + i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < input_offset + spatial_size
        elements = tl.load(x_ptr + offsets, mask=mask, other=0.0)
        sum_val += tl.sum(elements)
    
    # Compute the mean
    spatial_mean = sum_val / spatial_size
    
    # Store the result at the output position
    output_offset = pid
    tl.store(out_ptr + output_offset, spatial_mean)

# Kernel wrapper
@torch.fx.wrap
def fused_global_pool_flatten(x):
    # Get input shape
    batch_size, channels, height, width = x.shape
    
    # Calculate output size: batch_size * channels
    output_size = batch_size * channels
    
    # Calculate spatial size (height * width)
    spatial_size = height * width
    
    # Create output tensor
    out = torch.empty(output_size, dtype=torch.float32, device=x.device)
    
    # Set up grid and launch kernel
    grid = (output_size,)
    fused_global_pool_flatten_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        n_batch=batch_size,
        n_channels=channels,
        height=height,
        width=width,
        spatial_size=spatial_size,
        BLOCK_SIZE=1024
    )
    
    return out

# Replacement function (returns function reference)
def replacement_func():
    return fused_global_pool_flatten