import torch
import triton
import triton.language as tl

# Pattern matching function for mean reduction over spatial dimensions
def pattern(x):
    return x.mean((2, 3), keepdim=True)

# Argument extraction function
def replacement_args(x):
    return (x,)

# Optimized mean reduction kernel
@triton.jit
def mean_reduction_kernel(
    output_ptr,
    input_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr = 64,
):
    # Each program handles reduction for one channel across all batches
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    
    # Calculate starting position in input tensor
    start_offset = (batch_id * channels * height * width + 
                   channel_id * height * width)
    
    # Initialize sum for this batch and channel
    spatial_sum = 0.0
    
    # Iterate over spatial dimensions using loops
    for h in range(height):
        for w in range(width):
            offset = start_offset + h * width + w
            val = tl.load(input_ptr + offset)
            spatial_sum += val
    
    # Calculate mean
    mean_val = spatial_sum / (height * width)
    
    # Store result - output shape [batch_size, channels, 1, 1]
    output_idx = batch_id * channels + channel_id
    tl.store(output_ptr + output_idx, mean_val)

@torch.fx.wrap
def optimized_mean_reduction(x):
    batch_size, channels, height, width = x.shape
    
    # Flatten spatial dimensions for reduction
    # Each program handles one (batch, channel) pair
    total_pairs = batch_size * channels
    BLOCK_SIZE = 64
    
    # Calculate grid size
    num_pairs = (total_pairs + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Output shape: [batch_size, channels, 1, 1] flattened for easier access
    output = torch.empty(batch_size, channels, 1, 1, 
                        dtype=torch.float32, device=x.device)
    output_flat = output.view(-1)  # Flatten to [batch_size * channels]
    
    mean_reduction_kernel[(num_pairs,)](
        output_ptr=output_flat,
        input_ptr=x,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE_X=BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    return optimized_mean_reduction