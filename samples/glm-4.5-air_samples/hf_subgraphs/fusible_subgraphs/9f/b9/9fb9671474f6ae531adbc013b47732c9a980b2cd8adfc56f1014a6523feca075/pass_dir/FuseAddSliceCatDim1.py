import torch
import triton
import triton.language as tl

def pattern(x, y, z):
    """Pattern: addition + slicing + concatenation"""
    # Element-wise addition
    tmp_0 = x + y
    # Slice the third input tensor along channel dimension from specific start index
    tmp_1 = z[slice(None, None, None), slice(None, None, None)]
    # Concatenate along dimension 1 (channels)
    out = torch.cat([tmp_0, tmp_1], dim=1)
    return out

def replacement_args(x, y, z):
    """Extract arguments for the replacement kernel"""
    return x, y, z

@triton.jit
def fused_add_slice_cat_kernel(
    x_ptr,
    y_ptr,
    z_ptr,
    out_ptr,
    x_batch,
    x_channels,
    x_height,
    x_width,
    z_total_channels,
    slice_start_idx,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel: addition + slicing + concatenation"""
    # Compute total grid dimensions
    total_elements = x_batch * x_channels * x_height * x_width
    
    # Each program handles a contiguous block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Compute element indices
    batch_idx = offsets // (x_channels * x_height * x_width)
    remaining = offsets % (x_channels * x_height * x_width)
    channel_idx = remaining // (x_height * x_width)
    height_idx = (remaining % (x_height * x_width)) // x_width
    width_idx = remaining % x_width
    
    # Load input tensors
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform element-wise addition
    add_result = x + y
    
    # For the sliced tensor, we need to handle the offset
    z_slice_size = z_total_channels - slice_start_idx
    z_slice_elements = x_batch * z_slice_size * x_height * x_width
    
    # Compute z tensor indices for the slice
    z_total_elements_per_batch = z_total_channels * x_height * x_width
    z_slice_start_offset = slice_start_idx * x_height * x_width
    z_slice_offsets = batch_idx * z_total_elements_per_batch + z_slice_start_offset + height_idx * x_width + width_idx
    
    z_slice_mask = z_slice_offsets < (batch_idx * z_total_elements_per_batch + z_total_elements_per_batch)
    z_slice_mask = z_slice_mask & (channel_idx >= 0) & (channel_idx < z_slice_size)
    
    z_slice = tl.load(z_ptr + z_slice_offsets, mask=z_slice_mask, other=0.0)
    
    # Create output tensor
    out_channels = x_channels + z_slice_size
    out_elements = x_batch * out_channels * x_height * x_width
    out_offset_for_first = batch_idx * out_channels * x_height * x_width + channel_idx * x_height * x_width + height_idx * x_width + width_idx
    out_offset_for_second = batch_idx * out_channels * x_height * x_width + (x_channels + z_slice_size - 1) * x_height * x_width + height_idx * x_width + width_idx
    
    # For the addition result (first part of output)
    first_part_mask = channel_idx < x_channels
    tl.store(out_ptr + out_offset_for_first, add_result, mask=first_part_mask & mask)
    
    # For the slice result (second part of output)  
    second_part_mask = (channel_idx >= x_channels) & (channel_idx < out_channels)
    tl.store(out_ptr + out_offset_for_second, z_slice, mask=second_part_mask & mask & z_slice_mask)

@torch.fx.wrap
def fused_add_slice_cat_op(x, y, z):
    """Wrapper function for the fused operation"""
    # Get tensor shapes
    batch, x_channels, height, width = x.shape
    z_total_channels = z.shape[1]
    slice_start_idx = z_total_channels - (z.shape[1] - x_channels)  # This is approximate
    
    # Calculate output shape
    out_channels = x_channels + (z_total_channels - slice_start_idx)
    x_size = x.numel()
    
    # Create output tensor
    out = torch.empty((batch, out_channels, height, width), dtype=x.dtype, device=x.device)
    
    # Block size tuning
    if x_size < 1024:
        BLOCK_SIZE = 64
    elif x_size < 10000:
        BLOCK_SIZE = 256
    else:
        BLOCK_SIZE = 1024
    
    num_programs = (x_size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_add_slice_cat_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        z_ptr=z,
        out_ptr=out,
        x_batch=batch,
        x_channels=x_channels,
        x_height=height,
        x_width=width,
        z_total_channels=z_total_channels,
        slice_start_idx=slice_start_idx,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return fused_add_slice_cat_op