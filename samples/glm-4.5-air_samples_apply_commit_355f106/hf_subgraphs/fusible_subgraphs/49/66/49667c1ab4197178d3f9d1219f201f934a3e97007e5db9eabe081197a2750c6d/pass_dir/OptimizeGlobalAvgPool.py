import torch
import triton
import triton.language as tl

# Pattern matching function: Optimized AdaptiveAvgPool2d to 1x1 (Global Average Pooling)
def pattern(x):
    # This matches adaptive_avg_pool2d with output_size=(1, 1)
    out = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    return out

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized kernel for global average pooling
@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    # Each program computes one element of the output
    b = tl.program_id(0)
    c = tl.program_id(1)
    
    # Output element (since output is 1x1, we have one program per batch and channel)
    out_idx = b * channels + c
    out_ptr_idx = out_idx * 2  # Output is [batch_size, channels, 1, 1] -> flattened
    
    # Initialize accumulator
    sum_val = 0.0
    
    # Process spatial dimensions with tiling
    for h in range(0, height, BLOCK_SIZE_M):
        for w in range(0, width, BLOCK_SIZE_N):
            # Compute spatial bounds
            h_idx = h + tl.arange(0, BLOCK_SIZE_M)
            w_idx = w + tl.arange(0, BLOCK_SIZE_N)
            
            # Create mask for valid spatial indices
            h_mask = h_idx < height
            w_mask = w_idx < width
            
            # Compute global 2D mask
            mask = h_mask[:, None] & w_mask[None, :]
            
            # Flatten indices
            h_flat = h_idx[:, None] * width + w_idx[None, :]
            ptr_base = (b * channels + c) * height * width
            x_indices = ptr_base + h_flat
            
            # Load input elements and sum
            vals = tl.load(x_ptr + x_indices, mask=mask, other=0.0)
            sum_val += tl.sum(vals)
    
    # Compute average
    n_elements = height * width
    avg_val = sum_val / n_elements
    
    # Store output
    out_indices = [out_idx, 0, 0, 0]  # [batch, channel, 1, 1] layout
    tl.store(out_ptr + out_idx, avg_val)

@torch.fx.wrap
def optimized_global_avg_pool(x):
    # Get input dimensions
    batch_size, channels, height, width = x.shape
    
    # Output will be [batch_size, channels, 1, 1]
    out_shape = (batch_size, channels)
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Choose block sizes based on input dimensions
    if height <= 8 and width <= 8:
        BLOCK_SIZE_M = 8
        BLOCK_SIZE_N = 8
    elif height <= 16 and width <= 16:
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
    else:
        BLOCK_SIZE_M = 32
        BLOCK_SIZE_N = 32
    
    # Launch kernel
    grid = (batch_size, channels)
    global_avg_pool_kernel[grid](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return out

# Replacement function
def replacement_func():
    return optimized_global_avg_pool