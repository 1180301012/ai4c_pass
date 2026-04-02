import torch
import triton
import triton.language as tl

# Pattern matching function - matches the padded output
def pattern(tmp_4):
    """Match the padding operation: pad by (0, 1, 0, 1)"""
    tmp_5 = torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    return tmp_5

# Argument extraction function
def replacement_args(tmp_4):
    return (tmp_4,)

# Triton kernel optimized for right-bottom padding operation
@triton.jit
def pad_right_bottom_kernel(
    x_ptr,                      # Input tensor [B, C, H, W]
    output_ptr,                 # Output tensor [B, C, H+1, W+1] 
    n_elements_hw,              # Number of elements in H x W
    batch_size,                 # Batch size
    channels,                   # Number of channels
    height,                     # Input height  
    width,                      # Input width
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a 2D tile in H and W dimensions
    pid_h = tl.program_id(1)  # Height programs
    pid_w = tl.program_id(2)  # Width programs  
    pid_b = tl.program_id(0)  # Batch programs
    
    # 2D indexing within each tile
    offsets_h = pid_h * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offsets_w = pid_w * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Create masks for 2D bounds
    mask_h = offsets_h < height
    mask_w = offsets_w < width
    mask = mask_h[:, None] & mask_w[None, :]  # 2D mask
    
    # 2D coordinate arrays
    h_coords = offsets_h[:, None]
    w_coords = offsets_w[None, :]
    
    # Calculate input and output pointers
    # Base offset for current batch and channel  
    base_offset = pid_b * channels * height * width + 0 * height * width  # channel=0 for now
    base_offset = base_offset + tl.arange(channels)[:, None, None] * height * width
    base_offset = base_offset + h_coords[None, :, None] * width + w_coords[None, None, :]
    
    # Load input data (only for regions that exist)
    x = tl.load(x_ptr + base_offset, mask=mask[None, :, :], other=0.0)
    
    # Create output by replicating edge values for padding
    # Right padding: duplicate last column
    right_padding = w_coords == (width - 1)
    # Bottom padding: duplicate last row  
    bottom_padding = h_coords == (height - 1)
    
    # For padded regions, copy from edge values
    # This replicates the behavior of normal padding (constant None typically means copy edge)
    
    # Create padded output
    # Shape: [batch, channels, height+1, width+1]
    padded_output = tl.zeros((height + 1, width + 1), dtype=tl.float32)
    
    # Copy original data (interior region)
    padded_output[0:height, 0:width] = x
    
    # Right padding: duplicate last column
    if width > 0:
        padded_output[0:height, width:width+1] = x[:, :, :, width-1:width]
    
    # Bottom padding: duplicate last row  
    if height > 0:
        padded_output[height:height+1, 0:width] = x[:, :, height-1:height, :]
        
    # Bottom-right corner: use bottom edge value
    if height > 0 and width > 0:
        padded_output[height:height+1, width:width+1] = x[:, :, height-1:height, width-1:width]
    
    # Store output
    output_h_coords = h_coords
    output_w_coords = w_coords + 1  # Account for padding
    
    # Create 3D mask for output
    output_mask_h = output_h_coords < (height + 1)
    output_mask_w = output_w_coords < (width + 1)
    output_mask = output_mask_h[:, None] & output_mask_w[None, :]
    
    # Base offset for output
    output_base_offset = pid_b * channels * (height + 1) * (width + 1) + 0 * (height + 1) * (width + 1)
    output_base_offset = output_base_offset + tl.arange(channels)[:, None, None] * (height + 1) * (width + 1)
    output_base_offset = output_base_offset + output_h_coords[None, :, None] * (width + 1) + output_w_coords[None, None, :]
    
    tl.store(output_ptr + output_base_offset, padded_output[None, :, :], mask=output_mask[None, :, :])

# Optimized padding function
@torch.fx.wrap  
def optimized_pad_right_bottom(tmp_4):
    if tmp_4.dim() != 4:
        return torch.nn.functional.pad(tmp_4, (0, 1, 0, 1), 'constant', None)
    
    # Get tensor shape
    batch_size, channels, height, width = tmp_4.shape
    
    # Choose optimal block size for GPU occupancy
    BLOCK_SIZE = 16  # Smaller blocks for better 2D parallelism
    
    # Calculate grid dimensions
    num_batches = batch_size
    num_height = (height + BLOCK_SIZE - 1) // BLOCK_SIZE
    num_width = (width + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor with increased dimensions
    output_shape = (batch_size, channels, height + 1, width + 1)
    output = torch.empty(output_shape, dtype=tmp_4.dtype, device=tmp_4.device)
    
    # Launch the Triton kernel
    pad_right_bottom_kernel[(num_batches, num_height, num_width)](
        x_ptr=tmp_4,
        output_ptr=output,
        n_elements_hw=height * width,
        batch_size=batch_size,
        channels=channels,
        height=height,
        width=width,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

# Replacement function (returns function reference)
def replacement_func():
    return optimized_pad_right_bottom