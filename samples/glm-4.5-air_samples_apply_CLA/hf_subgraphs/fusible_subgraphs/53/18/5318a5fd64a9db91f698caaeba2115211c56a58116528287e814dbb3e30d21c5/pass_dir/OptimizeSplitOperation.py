import torch
import triton
import triton.language as tl

# Pattern matching function for split operation
def pattern(tmp_4):
    tmp_5 = torch.functional.split(tmp_4, [32, 48, 48], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return tmp_6, tmp_7, tmp_8

# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)

# Optimized Triton kernel for splitting tensor along channel dimension
@triton.jit
def split_kernel(
    input_ptr,
    output_1_ptr,
    output_2_ptr,
    output_3_ptr,
    batch_size,
    total_channels,    # Total channels (128 in our case)
    height,
    width,
    size_1,            # Size of first split (32)
    size_2,            # Size of second split (48)
    size_3,            # Size of third split (48)
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate total elements per channel
    elements_per_channel = height * width
    
    # Create block offsets
    channel_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Check bounds for each output tensor
    mask_1 = channel_offsets < size_1
    mask_2 = (channel_offsets >= size_1) & (channel_offsets < size_1 + size_2)
    mask_3 = (channel_offsets >= size_1 + size_2) & (channel_offsets < total_channels)
    
    # Load input data
    global_offsets = channel_offsets * elements_per_channel
    input_data = tl.load(input_ptr + global_offsets, mask=channel_offsets < total_channels, other=0.0)
    
    # Store to output 1 (first size_1 channels)
    output_1_offsets = (channel_offsets * elements_per_channel) if pid == 0 else global_offsets
    if pid == 0:  # Only first program handles output 1
        tl.store(output_1_ptr + (channel_offsets * elements_per_channel), input_data, mask=mask_1)
    
    # Store to output 2 (next size_2 channels)
    output_2_offsets = (channel_offsets - size_1) * elements_per_channel
    tl.store(output_2_ptr + output_2_offsets, input_data, mask=mask_2)
    
    # Store to output 3 (remaining size_3 channels)  
    output_3_offsets = (channel_offsets - size_1 - size_2) * elements_per_channel
    tl.store(output_3_ptr + output_3_offsets, input_data, mask=mask_3)

# Simple and efficient kernel for direct split operation
@triton.jit
def simple_split_kernel(
    input_ptr,
    output_1_ptr,
    output_2_ptr,
    output_3_ptr,
    batch_size,
    total_channels,
    height,
    width,
    size_1,
    size_2,
    size_3,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Process in blocks for efficiency
    channel_offset = pid * BLOCK_SIZE
    channel_end = min(channel_offset + BLOCK_SIZE, total_channels)
    
    # Calculate which output tensor this block belongs to
    if channel_offset >= size_1 + size_2 + size_3:
        return
    
    # Process each channel position
    for c in range(channel_offset, channel_end):
        if c < size_1:
            # This belongs to output 1
            out_1_c = c
            out_2_c = -1  # not used
            out_3_c = -1  # not used
        elif c < size_1 + size_2:
            # This belongs to output 2
            out_1_c = -1
            out_2_c = c - size_1
            out_3_c = -1
        else:
            # This belongs to output 3
            out_1_c = -1
            out_2_c = -1
            out_3_c = c - size_1 - size_2
        
        # Process spatial positions (flattened for efficiency)
        spatial_size = height * width
        spatial_start = 0
        
        # Process spatial positions in blocks
        spatial_block_size = 1024
        spatial_blocks = (spatial_size + spatial_block_size - 1) // spatial_block_size
        spatial_grid_id = tl.program_id(1)
        
        if spatial_grid_id >= spatial_blocks:
            continue
            
        spatial_start_idx = spatial_grid_id * spatial_block_size
        spatial_end_idx = min(spatial_start_idx + spatial_block_size, spatial_size)
        
        for s in range(spatial_start_idx, spatial_end_idx):
            # Global input offset
            input_offset = (batch_size * total_channels + c) * spatial_size + s
            
            # Store to appropriate output
            if out_1_c >= 0:
                output_1_offset = (batch_size * size_1 + out_1_c) * spatial_size + s
                data = tl.load(input_ptr + input_offset, other=0.0)
                tl.store(output_1_ptr + output_1_offset, data)
            elif out_2_c >= 0:
                output_2_offset = (batch_size * size_2 + out_2_c) * spatial_size + s
                data = tl.load(input_ptr + input_offset, other=0.0)
                tl.store(output_2_ptr + output_2_offset, data)
            elif out_3_c >= 0:
                output_3_offset = (batch_size * size_3 + out_3_c) * spatial_size + s
                data = tl.load(input_ptr + input_offset, other=0.0)
                tl.store(output_3_ptr + output_3_offset, data)

# Kernel wrapper (MUST be decorated with @torch.fx.wrap)
@torch.fx.wrap
def direct_split_triton(input_tensor, size_1, size_2, size_3):
    # Get input tensor shape
    batch_size, total_channels, height, width = input_tensor.shape
    
    # Verify total channels match the split sizes
    assert total_channels == size_1 + size_2 + size_3, f"Total channels {total_channels} doesn't match {size_1}+{size_2}+{size_3}"
    
    # Create output tensors
    output_1 = torch.empty((batch_size, size_1, height, width), dtype=torch.float32, device=input_tensor.device)
    output_2 = torch.empty((batch_size, size_2, height, width), dtype=torch.float32, device=input_tensor.device)
    output_3 = torch.empty((batch_size, size_3, height, width), dtype=torch.float32, device=input_tensor.device)
    
    # Flatten tensors for simpler indexing
    input_flat = input_tensor.reshape(batch_size, -1)
    output_1_flat = output_1.reshape(batch_size, -1)
    output_2_flat = output_2.reshape(batch_size, -1) 
    output_3_flat = output_3.reshape(batch_size, -1)
    
    # Set up grid dimensions
    total_channels_grid = (total_channels + 255) // 256  # Process channels in blocks of 256
    spatial_size = height * width
    spatial_grid_size = (spatial_size + 1023) // 1024  # Process spatial in blocks
    
    # Launch simplified kernel with 2D grid
    simple_split_kernel[(total_channels_grid, spatial_grid_size)](
        input_flat,
        output_1_flat,
        output_2_flat,
        output_3_flat,
        batch_size,
        total_channels,
        height,
        width,
        size_1,
        size_2,
        size_3,
        BLOCK_SIZE=256  # Process 256 channels per block
    )
    
    return output_1, output_2, output_3

# Helper function to pass split sizes
def create_split_func(size_1, size_2, size_3):
    def split_func(tmp_4):
        return direct_split_triton(tmp_4, size_1, size_2, size_3)
    return split_func

# Replacement function (NO arguments, returns function reference)
def replacement_func():
    # Return split function with specific sizes for the first graph
    return create_split_func(32, 48, 48)