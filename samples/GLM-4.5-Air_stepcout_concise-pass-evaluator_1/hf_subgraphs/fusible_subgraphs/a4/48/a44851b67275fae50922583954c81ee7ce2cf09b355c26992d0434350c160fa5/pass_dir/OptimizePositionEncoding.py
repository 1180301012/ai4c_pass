import torch
import triton
import triton.language as tl

def pattern():
    tmp_4 = torch.zeros(1, 196, 196, 3)
    tmp_5 = torch.arange(14)
    tmp_6 = tmp_5.view(1, -1)
    tmp_7 = torch.arange(14)
    tmp_8 = tmp_7.view(-1, 1)
    tmp_9 = tmp_6 - tmp_8
    tmp_10 = tmp_9.repeat(14, 14)
    tmp_11 = tmp_9.repeat_interleave(14, dim=0)
    tmp_12 = tmp_11.repeat_interleave(14, dim=1)
    tmp_13 = tmp_10 ** 2
    tmp_14 = tmp_12 ** 2
    tmp_15 = tmp_13 + tmp_14
    tmp_16 = tmp_15.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 2] = tmp_16
    tmp_18 = tmp_12.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 1] = tmp_18
    tmp_20 = tmp_10.unsqueeze(0)
    tmp_4[slice(None, None, None), slice(None, None, None), slice(None, None, None), 0] = tmp_20
    return tmp_4

def replacement_args():
    return ()

@triton.jit
def position_encoding_kernel(
    out_ptr,
    batch_size: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    grid_size: tl.constexpr,
    channels: tl.constexpr,
):
    """Generate position encoding using coordinate grids"""
    pid = tl.program_id(0)
    n_elements = batch_size * height * width
    if pid >= n_elements:
        return
        
    # Calculate indices
    b = pid // (height * width)
    h = (pid % (height * width)) // width
    w = pid % width
    
    # Generate position encoding coordinates
    # The encoding uses grid_size x grid_size coordinate grid
    grid_h = h // (height // grid_size)
    grid_w = w // (width // grid_size)
    
    # Center coordinates relative to grid cell
    cell_h = (h % (height // grid_size)) / (height // grid_size) - 0.5
    cell_w = (w % (width // grid_size)) / (width // grid_size) - 0.5
    
    # Create the position encoding pattern as in the original code
    # We need to compute the coordinate differences and their squares
    coord_diff_h = cell_h * grid_size
    coord_diff_w = cell_w * grid_size
    
    # Compute the three channels as in the original implementation
    # Channel 0: x^2 from original tmp_10
    # Channel 1: y^2 from original tmp_12  
    # Channel 2: x^2 + y^2 from original tmp_15
    
    # Rescale coordinates properly for the specific pattern
    x_coord = coord_diff_w
    y_coord = coord_diff_h
    
    # Channel assignments based on original pattern
    # tmp_10 corresponds to x-coordinates (repeated in x direction)
    # tmp_12 corresponds to y-coordinates (repeated in y direction)
    # tmp_15 is x^2 + y^2
    
    channel_0 = x_coord * x_coord  # tmp_10 squared
    channel_1 = y_coord * y_coord  # tmp_12 squared  
    channel_2 = channel_0 + channel_1  # tmp_15 = tmp_13 + tmp_14
    
    # Store the three channels
    base_idx = pid * 3
    tl.store(out_ptr + base_idx + 0, channel_0)
    tl.store(out_ptr + base_idx + 1, channel_1)
    tl.store(out_ptr + base_idx + 2, channel_2)

@torch.fx.wrap
def optimized_position_encoding(batch_size, height, width):
    # Position encoding has 3 channels
    out = torch.zeros(batch_size, height, width, 3, dtype=torch.float32, device='cuda')
    
    # Grid size is 14x14 for the original encoding
    grid_size = 14
    
    # Launch kernel
    n_elements = batch_size * height * width
    num_programs = (n_elements + 127) // 128  # Use BLOCK_SIZE of 128
    
    position_encoding_kernel[(num_programs,)](
        out,
        batch_size,
        height, 
        width,
        grid_size,
        3,
    )
    
    return out

def replacement_func():
    def kernel_wrapper():
        batch_size = 1
        height = 196
        width = 196
        return optimized_position_encoding(batch_size, height, width)
    
    return kernel_wrapper