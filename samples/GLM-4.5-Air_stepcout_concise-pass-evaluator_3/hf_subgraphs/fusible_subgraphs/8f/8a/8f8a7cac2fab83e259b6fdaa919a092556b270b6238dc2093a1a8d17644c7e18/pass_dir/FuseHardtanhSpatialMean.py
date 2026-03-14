import torch
import triton
import triton.language as tl

# Pattern matching function - simple test just matching adaptive_avg_pool2d
def pattern(in_0):
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(in_0, (1, 1))
    return (tmp_1,)

# Argument extraction function
def replacement_args(in_0):
    return (in_0,)

def next_power_of_2(n):
    """Compute the next power of 2 >= n"""
    if n == 0:
        return 1
    p = 1
    while p < n:
        p *= 2
    return p

# Optimized Triton kernel for fused hardtanh + spatial mean
@triton.jit
def fused_hardtanh_spatial_mean_kernel(
    x_ptr,          # Input tensor pointer
    out_ptr,        # Output tensor pointer  
    n_batch,        # Batch size (N)
    n_channels,     # Number of channels (C)
    height,         # Input height (H)
    width,          # Input width (W)
    SPATIAL_SIZE: tl.constexpr,  # Spatial size H*W as compile-time constant
    NEXT_POWER_OF_2: tl.constexpr,  # Next power of 2 >= spatial_size
    BLOCK_SIZE: tl.constexpr,  # Channel block size
):
    # Each program handles a batch dimension and multiple channels
    batch_idx = tl.program_id(0)
    channel_block_idx = tl.program_id(1)
    
    # Calculate channel range for this block
    channel_start = channel_block_idx * BLOCK_SIZE
    channel_end = min(channel_start + BLOCK_SIZE, n_channels)
    
    # Check if this block has any channels to process
    if channel_start >= n_channels:
        return
    
    # Process each channel in the block
    for channel_idx in range(channel_start, channel_end):
        # Calculate global memory offset for this batch and channel
        batch_channel_offset = batch_idx * n_channels * SPATIAL_SIZE + channel_idx * SPATIAL_SIZE
        
        # Load input data with proper masking for non-power-of-2 sizes
        spatial_offsets = tl.arange(0, NEXT_POWER_OF_2)
        input_offsets = batch_channel_offset + spatial_offsets
        
        # Create mask to ensure we don't go out of bounds
        mask = spatial_offsets < SPATIAL_SIZE
        
        # Load input with mask
        input_data = tl.load(x_ptr + input_offsets, mask=mask, other=0.0)
        
        # Apply hardtanh: max(0.0, min(6.0, x))
        clamped = tl.where(input_data < 0.0, 0.0, 
                          tl.where(input_data > 6.0, 6.0, input_data))
        
        # Compute spatial mean by summing and dividing by spatial size
        spatial_sum = tl.sum(clamped)
        spatial_mean = spatial_sum / SPATIAL_SIZE
        
        # Calculate output offset
        output_offset = batch_idx * n_channels + channel_idx
        
        # Store result
        tl.store(out_ptr + output_offset, spatial_mean)

# Optimized kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def fused_hardtanh_spatial_mean(in_0):
    # Get input tensor properties
    batch_size, channels, height, width = in_0.shape
    output_shape = (batch_size, channels)
    
    # Create output tensor
    output = torch.empty(output_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Calculate spatial size and next power of 2 as compile-time constants
    spatial_size = height * width
    next_power_of_2_size = next_power_of_2(spatial_size)
    
    # Optimized block size selection for maximum GPU occupancy
    # Choose block size based on tensor characteristics for optimal performance
    if spatial_size <= 64:  # Small spatial dimensions
        if batch_size <= 4 and channels <= 256:
            BLOCK_SIZE = 256  # Fewer programs, higher occupancy for small tensors
        else:
            BLOCK_SIZE = 64   # Balance for medium tensors
    else:  # Large spatial dimensions
        if channels <= 128:
            BLOCK_SIZE = 128  # Larger blocks for fewer total programs
        else:
            BLOCK_SIZE = 64   # Conservative block size for complex workloads
    
    # Calculate grid configuration - each program handles one batch and a block of channels
    num_channel_blocks = (channels + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid = (batch_size, num_channel_blocks)
    
    # Launch kernel with optimized constants
    fused_hardtanh_spatial_mean_kernel[grid](
        x_ptr=in_0,
        out_ptr=output,
        n_batch=batch_size,
        n_channels=channels, 
        height=height,
        width=width,
        SPATIAL_SIZE=spatial_size,
        NEXT_POWER_OF_2=next_power_of_2_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_hardtanh_spatial_mean