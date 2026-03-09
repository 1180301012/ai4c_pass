import torch
import triton
import triton.language as tl

def pattern(x):
    return torch.nn.functional.adaptive_avg_pool2d(x, 1)

def replacement_args(x):
    return (x,)

def get_hw_grid_size(h, w):
    """Returns grid dimensions for spatial dimensions"""
    # Use a reasonable number of programs per dim for better GPU utilization
    return max(1, h // 4), max(1, w // 4)

@triton.jit
def global_avg_pool_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program computes one output element (one batch, one channel)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # Total number of spatial elements for this (batch, channel) pair
    total_elements = height * width
    
    # Initialize accumulator
    accumulator = 0.0
    elements_processed = 0
    
    # Process spatial dimensions in blocks
    for h in range(0, height, BLOCK_SIZE):
        for w in range(0, width, BLOCK_SIZE):
            # Calculate block boundaries
            h_end = min(h + BLOCK_SIZE, height)
            w_end = min(w + BLOCK_SIZE, width)
            
            # Process elements in this block
            for hi in range(h, h_end):
                for wi in range(w, w_end):
                    # Calculate input offset for (batch, channel, hi, wi)
                    input_offset = (pid_b * channels + pid_c) * height * width + hi * width + wi
                    accumulator += tl.load(x_ptr + input_offset)
                    elements_processed += 1
    
    # Compute average (handle edge cases where we processed fewer elements)
    if elements_processed > 0:
        result = accumulator / elements_processed
    else:
        result = 0.0
    
    # Calculate output offset (output is [batch, channels, 1, 1])
    out_offset = (pid_b * channels + pid_c) * 1 * 1
    
    # Store result
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def optimized_global_avg_pool(x):
    batch_size, channels, height, width = x.shape
    
    # Calculate output shape: [batch_size, channels, 1, 1]
    output_shape = (batch_size, channels, 1, 1)
    out = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Set grid dimensions (one program per batch channel pair)
    grid_b = batch_size
    grid_c = channels
    
    # Choose optimal block size for spatial dimensions
    BLOCK_SIZE = 32  # Good balance between memory and compute
    
    # Launch kernel
    global_avg_pool_kernel[(grid_b, grid_c)](
        x_ptr=x,
        out_ptr=out,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_global_avg_pool