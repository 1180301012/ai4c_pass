import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matches the fused add + slice + concat operations.
    This handles Pattern A: in_0 + in_1, slice in_2, then concatenate.
    
    Based on the provided examples, using slice(210, None, None) which appears
    in rexnet_150.nav_in1k model.
    """
    tmp_0 = in_0 + in_1
    tmp_1 = in_2[slice(None, None, None), slice(210, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return (tmp_2,)

def replacement_args(in_0, in_1, in_2):
    """
    Extract arguments for the replacement function.
    Since we hardcoded slice(210, None, None) in the pattern, we use 210 as the slice index.
    """
    slice_start_idx = 210  # Matches the hardcoded pattern
    return (in_0, in_1, in_2, slice_start_idx)

@triton.jit
def fused_add_slice_concat_kernel(
    # Input tensors
    in_0_ptr,
    in_1_ptr, 
    in_2_ptr,
    # Output tensor
    out_ptr,
    # Input shapes
    batch_size,
    channels_0,
    channels_1,
    channels_2,
    height,
    width,
    slice_start_idx,
    # Config
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for: (in_0 + in_1) concatenated with in_2[:, slice_start_idx:, :, :]
    Uses 3D grid: (batch, spatial_idx, channel)
    """
    pid = tl.program_id(0)
    
    # Calculate total elements per batch and total output channels
    out_channels = channels_0 + (channels_2 - slice_start_idx)
    elements_per_batch = out_channels * height * width
    elements_per_spatial_loc = out_channels
    
    # Calculate batch and spatial location
    batch_idx = pid // (height * width * elements_per_spatial_loc)
    spatial_idx = pid % (height * width)
    spatial_pid = pid // elements_per_spatial_loc
    channel_idx = pid % elements_per_spatial_loc
    
    # Calculate spatial coordinates
    h_idx = spatial_idx // width
    w_idx = spatial_idx % width
    
    # Skip if out of bounds
    if batch_idx >= batch_size or (h_idx >= height or w_idx >= width):
        return
    
    # Determine which part of the output this element comes from
    if channel_idx < channels_0:
        # This is from the (in_0 + in_1) part
        c = channel_idx
        
        # Load from in_0 and in_1, then add
        offset_0 = (batch_idx * channels_0 + c) * height * width + h_idx * width + w_idx
        offset_1 = (batch_idx * channels_1 + c) * height * width + h_idx * width + w_idx
        
        val_0 = tl.load(in_0_ptr + offset_0)
        val_1 = tl.load(in_1_ptr + offset_1)
        result = val_0 + val_1
        
    else:
        # This is from the sliced in_2 part
        c_2_orig = slice_start_idx + (channel_idx - channels_0)
        
        # Only process if the original channel index is valid
        if c_2_orig < channels_2:
            offset_2 = (batch_idx * channels_2 + c_2_orig) * height * width + h_idx * width + w_idx
            result = tl.load(in_2_ptr + offset_2)
        else:
            result = 0.0
    
    # Store the result
    out_offset = (batch_idx * out_channels + channel_idx) * height * width + h_idx * width + w_idx
    tl.store(out_ptr + out_offset, result)

@torch.fx.wrap
def fused_add_slice_concat_optimized(in_0, in_1, in_2, slice_start_idx=210):
    """
    Optimized fused kernel for Pattern A: (in_0 + in_1) concatenated with in_2[:, slice_start_idx:, :, :]
    
    Args:
        slice_start_idx: Starting index for slicing in_2 (default 210 for first discovered pattern)
    """
    # Get input shapes
    batch_size, channels_0, height, width = in_0.shape
    _, channels_1, _, _ = in_1.shape
    _, channels_2, _, _ = in_2.shape
    
    # Validate slice index
    if slice_start_idx >= channels_2:
        slice_start_idx = channels_2 - 1  # Ensure valid slice
    
    # Create output tensor
    output_channels = channels_0 + (channels_2 - slice_start_idx)
    out_shape = (batch_size, output_channels, height, width)
    output = torch.empty(out_shape, dtype=in_0.dtype, device=in_0.device)
    
    # Set up grid and launch kernel
    total_elements = batch_size * output_channels * height * width
    BLOCK_SIZE = 1024
    grid_size = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_add_slice_concat_kernel[(grid_size,)](
        in_0,
        in_1,
        in_2,
        output,
        batch_size,
        channels_0,
        channels_1,
        channels_2,
        height,
        width,
        slice_start_idx,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_add_slice_concat_optimized