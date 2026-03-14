import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2):
    """
    Pattern matches: addition + slice + concatenation along channel dimension
    This mirrors the exact computation structure from model.py
    """
    tmp_0 = in_0 + in_1
    slice_start_idx = in_2.shape[1] - in_1.shape[1]  # Calculate slice start index
    tmp_1 = in_2[slice(None, None, None), slice(slice_start_idx, None, None)]
    tmp_2 = torch.cat([tmp_0, tmp_1], dim=1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    # Calculate slice start index for the kernel
    slice_start_idx = in_2.shape[1] - in_1.shape[1]
    return in_0, in_1, in_2, slice_start_idx

@triton.jit
def fused_add_slice_concat_kernel(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    out_ptr,
    channel_0,
    channel_1,
    channel_2,
    slice_start_idx,
    height,
    width,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_elements = channel_0 * channel_1 * height * width
    
    # Each program handles a contiguous block of data
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_elements
    
    # Calculate indices for 4D tensor (B, C, H, W)
    # flatten B and C dimensions for coalesced memory access
    bc_dim = channel_0
    hw_dim = height * width
    
    # Process first part: in_0 + in_1 (channels [0:channel_0])
    bc_offset_first = (offsets // hw_dim) % bc_dim
    hw_offset = offsets % hw_dim
    h_offset = hw_offset // width
    w_offset = hw_offset % width
    
    in_0_first_idx = bc_offset_first * hw_dim + hw_offset
    in_1_first_idx = bc_offset_first * hw_dim + hw_offset
    
    in_0_data = tl.load(in_0_ptr + in_0_first_idx, mask=mask, other=0.0)
    in_1_data = tl.load(in_1_ptr + in_1_first_idx, mask=mask, other=0.0)
    add_result = in_0_data + in_1_data
    
    # Store first part
    first_out_idx = bc_offset_first * hw_dim + hw_offset
    tl.store(out_ptr + first_out_idx, add_result, mask=mask)
    
    # Process second part: slice from in_2 (channels [slice_start_idx:slice_start_idx+channel_1])
    # Continue where first part ended
    if channel_0 < bc_dim:
        bc_offset_second = bc_offset_first + channel_0
        mask_second = (bc_offset_second < bc_dim) & mask
        
        if slice_start_idx + channel_0 <= channel_2:
            # Handle cases where slice and addition don't overlap in memory
            for chunk_start in range(0, channel_1, channel_0):
                chunk_end = min(chunk_start + channel_0, channel_1)
                current_channel_offset = channel_0 + chunk_start
                
                if current_channel_offset < bc_dim:
                    mask_chunk = (current_channel_offset < bc_dim) & mask
                    
                    in_2_idx = (current_channel_offset) * hw_dim + hw_offset
                    slice_out_idx = bc_offset_first * hw_dim + current_channel_offset * hw_dim + hw_offset
                    
                    in_2_data = tl.load(in_2_ptr + in_2_idx, mask=mask_chunk, other=0.0)
                    tl.store(out_ptr + slice_out_idx, in_2_data, mask=mask_chunk)
        else:
            # Handle edge case where slice might wrap around
            for chunk_start in range(0, channel_1, channel_0):
                chunk_end = min(chunk_start + channel_0, channel_1)
                current_channel_in_output = channel_0 + chunk_start
                current_channel_in_input = slice_start_idx + chunk_start
                
                if current_channel_in_output < bc_dim and current_channel_in_input < channel_2:
                    mask_chunk = (current_channel_in_output < bc_dim) & mask
                    
                    in_2_idx = current_channel_in_input * hw_dim + hw_offset
                    slice_out_idx = bc_offset_first * hw_dim + current_channel_in_output * hw_dim + hw_offset
                    
                    in_2_data = tl.load(in_2_ptr + in_2_idx, mask=mask_chunk, other=0.0)
                    tl.store(out_ptr + slice_out_idx, in_2_data, mask=mask_chunk)

@torch.fx.wrap
def fused_add_slice_concat(in_0, in_1, in_2, slice_start_idx):
    """Fused kernel that performs addition + slice + concatenation"""
    # Get tensor shapes
    batch_size, channels_0, height, width = in_0.shape
    channels_1 = in_1.shape[1]
    channels_2 = in_2.shape[1]
    
    # Output channels = channels_0 (from addition) + channels_1 (from slice)  
    output_channels = channels_0 + channels_1
    
    # Calculate total number of elements
    total_elements = batch_size * output_channels * height * width
    
    # Choose block size for optimal GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty((batch_size, output_channels, height, width), 
                     dtype=in_0.dtype, device=in_0.device)
    
    # Launch the fused kernel
    fused_add_slice_concat_kernel[(num_programs,)](
        in_0_ptr=in_0,
        in_1_ptr=in_1,
        in_2_ptr=in_2,
        out_ptr=out,
        channel_0=channels_0,
        channel_1=channels_1,
        channel_2=channels_2,
        slice_start_idx=slice_start_idx,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return fused_add_slice_concat