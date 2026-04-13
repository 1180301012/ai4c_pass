import torch
import triton
import triton.language as tl

# Pattern matching function for adaptive average pooling
def pattern(tmp_1):
    """Match adaptive average pooling with output size 1"""
    tmp_2 = torch.nn.functional.adaptive_avg_pool2d(tmp_1, 1)
    return tmp_2

# Argument extraction function
def replacement_args(tmp_1):
    return (tmp_1,)

# Optimized kernel for adaptive average pooling to 1x1
@triton.jit
def adaptive_avg_pool_1x1_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    in_height,
    in_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output position (per channel per batch)
    pid = tl.program_id(0)
    batch_id = pid // channels
    channel_id = pid % channels
    
    # Calculate input start positions for this batch and channel
    batch_offset = batch_id * channels * in_height * in_width
    channel_offset = channel_id * in_height * in_width
    spatial_offset = batch_offset + channel_offset
    
    # Load all spatial elements for this batch and channel
    spatial_elements = tl.load(x_ptr + spatial_offset + tl.arange(0, in_height * in_width))
    
    # Compute mean (average) across spatial dimensions
    mean = tl.sum(spatial_elements) / (in_height * in_width)
    
    # Store result at the corresponding output position
    out_idx = batch_id * channels + channel_id
    tl.store(out_ptr + out_idx, mean)

@torch.fx.wrap
def optimized_adaptive_avg_pool_1x1(tmp_1):
    """Optimized adaptive average pooling to 1x1"""
    shape = tmp_1.shape
    batch_size, channels, in_height, in_width = shape[0], shape[1], shape[2], shape[3]
    
    # Output shape: [batch_size, channels, 1, 1]
    out_shape = (batch_size, channels, 1, 1)
    out = torch.empty(out_shape, dtype=tmp_1.dtype, device=tmp_1.device)
    
    # Flatten output for easier 1D kernel execution
    out_flat = out.view(-1)
    
    block_size = 1024  # Process multiple elements per thread
    total_elements = batch_size * channels
    num_programs = (total_elements + block_size - 1) // block_size
    
    if total_elements > 0:
        kernel = adaptive_avg_pool_1x1_kernel[(num_programs,)](
            x_ptr=tmp_1,
            out_ptr=out_flat,
            batch_size=batch_size,
            channels=channels,
            in_height=in_height,
            in_width=in_width,
            BLOCK_SIZE=block_size,
        )
    
    # Reshape back to 4D
    return out.view(out_shape)

# Replacement function
def replacement_func():
    return optimized_adaptive_avg_pool_1x1