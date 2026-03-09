import torch
import triton
import triton.language as tl

# Pattern matching function: Fuse ReLU + Add + Global Average Pooling
def pattern(x, y):
    relu_out = torch.nn.functional.relu(x, inplace=False)
    add_out = relu_out + y
    pooled_out = torch.nn.functional.adaptive_avg_pool2d(add_out, 1)
    # Only return what the original model returns - just the pooled output
    return pooled_out

# Argument extraction function
def replacement_args(x, y):
    return (x, y)

# Optimized kernel for fully fused ReLU + Add + Global Average Pooling
@triton.jit
def fused_relu_add_global_pool_kernel(
    x_ptr,
    y_ptr,
    pooled_out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_SPATIAL: tl.constexpr,
):
    # Each program handles one batch and channel combination
    b = tl.program_id(0)  # batch ID
    c = tl.program_id(1)  # channel ID
    
    # Global accumulator for pooling
    total_sum = 0.0
    
    # Process spatial dimensions in tiles
    for h in range(0, height, BLOCK_SIZE_SPATIAL):
        for w in range(0, width, BLOCK_SIZE_SPATIAL):
            # Compute spatial tile bounds
            h_bounds = h + tl.arange(0, BLOCK_SIZE_SPATIAL)
            w_bounds = w + tl.arange(0, BLOCK_SIZE_SPATIAL)
            
            # Mask for valid spatial indices
            h_mask = h_bounds < height
            w_mask = w_bounds < width
            spatial_mask = h_mask[:, None] & w_mask[None, :]
            
            # Flatten spatial indices
            spatial_flat = h_bounds[:, None] * width + w_bounds[None, :]
            ptr_base = (b * channels + c) * height * width
            indices = ptr_base + spatial_flat
            
            # Load input data
            x_vals = tl.load(x_ptr + indices, mask=spatial_mask, other=0.0)
            y_vals = tl.load(y_ptr + indices, mask=spatial_mask, other=0.0)
            
            # Fused operations: ReLU + Add
            relu_vals = tl.maximum(x_vals, 0.0)
            add_vals = relu_vals + y_vals
            
            # Accumulate for global average pooling
            total_sum += tl.sum(add_vals)
    
    # Compute global average
    n_spatial_elements = height * width
    pooled_val = total_sum / n_spatial_elements
    
    # Store final pooled output
    pooled_idx = b * channels + c
    tl.store(pooled_out_ptr + pooled_idx, pooled_val)

@torch.fx.wrap
def fused_relu_add_global_pool(x, y):
    batch_size, channels, height, width = x.shape
    
    # Allocate only the output that we need to return
    pooled_out = torch.empty((batch_size, channels), dtype=x.dtype, device=x.device)
    
    # Choose block sizes dynamically
    if height <= 8 and width <= 8:
        BLOCK_SIZE_SPATIAL = 8
    elif height <= 16 and width <= 16:
        BLOCK_SIZE_SPATIAL = 16
    elif height <= 32 and width <= 32:
        BLOCK_SIZE_SPATIAL = 32
    else:
        BLOCK_SIZE_SPATIAL = 64
    
    # Launch kernel - each batch and channel combination gets one program
    grid = (batch_size, channels)
    
    fused_relu_add_global_pool_kernel[grid](
        x_ptr=x,
        y_ptr=y,
        pooled_out_ptr=pooled_out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_SPATIAL=BLOCK_SIZE_SPATIAL,
    )
    
    return pooled_out

# Replacement function
def replacement_func():
    return fused_relu_add_global_pool