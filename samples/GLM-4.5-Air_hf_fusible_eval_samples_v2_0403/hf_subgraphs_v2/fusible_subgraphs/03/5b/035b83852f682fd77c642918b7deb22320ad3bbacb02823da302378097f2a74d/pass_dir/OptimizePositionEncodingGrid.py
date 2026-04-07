import torch
import triton
import triton.language as tl

def pattern():
    """
    Pattern matching function for position encoding computation.
    This matches the sequence of operations that creates relative position encodings:
    - Create coordinate grid using arange, view, subtraction
    - Expand grid using repeat operations  
    - Compute squared distances
    - Assign to different channels of output tensor
    """
    tmp_3 = torch.zeros(1, 196, 196, 3)
    tmp_4 = torch.arange(14)
    tmp_5 = tmp_4.view(1, -1)
    tmp_6 = torch.arange(14)
    tmp_7 = tmp_6.view(-1, 1)
    tmp_8 = tmp_5 - tmp_7
    tmp_9 = tmp_8.repeat(14, 14)
    tmp_10 = tmp_8.repeat_interleave(14, dim = 0)
    tmp_11 = tmp_10.repeat_interleave(14, dim = 1)
    tmp_12 = tmp_9 ** 2
    tmp_13 = tmp_11 ** 2
    tmp_14 = tmp_12 + tmp_13
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 2)] = tmp_15
    tmp_17 = tmp_11.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 1)] = tmp_17
    tmp_19 = tmp_9.unsqueeze(0)
    tmp_3[(slice(None, None, None), slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    return tmp_3

def replacement_args():
    """
    Return empty tuple since this pattern doesn't depend on external inputs.
    The pattern creates its own inputs internally.
    """
    return ()

@triton.jit
def position_encoding_kernel(
    output_ptr,
    n_rows: tl.constexpr,
    n_cols: tl.constexpr,
    n_channels: tl.constexpr,
    grid_size: tl.constexpr,
):
    """
    Triton kernel to compute relative position encoding in a single pass.
    This kernel replaces the multiple inefficient CPU operations with one GPU kernel.
    """
    # Calculate program ID and total grid size
    pid = tl.program_id(0)
    total_elements = n_rows * n_cols * n_channels
    block_size = 1024  # Optimal block size for modern GPUs
    
    # Calculate start index for this program
    start_idx = pid * block_size
    offsets = start_idx + tl.arange(0, block_size)
    mask = offsets < total_elements
    
    # Convert flat offset to 3D coordinates (batch, row, col, channel)
    offset_div = offsets
    batch_idx = offset_div // (n_rows * n_cols * n_channels)
    offset_div = offset_div % (n_rows * n_cols * n_channels)
    
    col_idx = offset_div // (n_rows * n_channels)
    offset_div = offset_div % (n_rows * n_channels)
    
    row_idx = offset_div // n_channels
    channel_idx = offset_div % n_channels
    
    # Get relative coordinates from grid coordinates
    # Convert to 0-based grid indices (0-13 for original grid, scaled up to 0-195)
    grid_col = col_idx % grid_size
    grid_row = row_idx % grid_size
    
    # Center the grid at zero and scale
    center_offset = grid_size // 2
    rel_col = grid_col - center_offset
    
    if channel_idx == 0:
        # Channel 0: original x coordinates (expanded)
        x_coord = rel_col
        y_coord = 0  # Will be added in channel 1
    elif channel_idx == 1:
        # Channel 1: original y coordinates (expanded)  
        x_coord = 0  # Will be added in channel 0
        y_coord = rel_col
    else:  # channel_idx == 2
        # Channel 2: squared distance (x^2 + y^2)
        x_coord = rel_col
        y_coord = rel_col
    
    # Assign values based on channel
    if channel_idx < 2:
        # For channels 0 and 1, assign the coordinate directly
        if channel_idx == 0:
            result = x_coord.float()  # Cast to float for consistency
        else:  # channel_idx == 1
            result = y_coord.float()
    else:  # channel_idx == 2
        # For channel 2, compute squared distance
        x_squared = x_coord.float() ** 2
        y_squared = y_coord.float() ** 2
        result = x_squared + y_squared
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def compute_position_encoding_optimized():
    """
    Optimized wrapper function that launches the Triton kernel for position encoding.
    This replaces the inefficient CPU-based coordinate computation.
    """
    # Output shape matches original: (1, 196, 196, 3)  
    output = torch.empty(1, 196, 196, 3, dtype=torch.float32, device='cuda')
    
    # Kernel launch parameters
    n_rows, n_cols, n_channels = 196, 196, 3
    grid_size = 14  # Original grid size (14x14)
    total_elements = n_rows * n_cols * n_channels
    block_size = 1024
    
    # Calculate number of programs needed
    num_programs = (total_elements + block_size - 1) // block_size
    
    # Launch the kernel
    position_encoding_kernel[(num_programs,)](
        output_ptr=output,
        n_rows=n_rows,
        n_cols=n_cols,
        n_channels=n_channels,
        grid_size=grid_size,
    )
    
    return output

def replacement_func():
    """
    Returns the optimized position encoding computation function.
    """
    return compute_position_encoding_optimized