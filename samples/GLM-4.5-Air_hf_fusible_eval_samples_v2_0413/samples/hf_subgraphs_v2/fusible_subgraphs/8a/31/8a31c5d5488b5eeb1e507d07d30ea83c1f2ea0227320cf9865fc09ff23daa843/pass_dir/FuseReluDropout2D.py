import torch
import triton
import triton.language as tl

def pattern(in_0):
    # Match the exact computation from model.py including cleanup statement
    tmp_0 = torch.nn.functional.relu(in_0, inplace = True)
    in_0 = None
    tmp_1 = torch.nn.functional.dropout2d(tmp_0, 0.1, False, False)
    return (tmp_1, tmp_0)

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def fused_relu_dropout_kernel(
    input_ptr,
    dropout_output_ptr,
    relu_output_ptr,
    batch_size,
    channels,
    height,
    width,
    dropout_ratio: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    # Batch and channel indices
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate input/output dimensions
    input_size = height * width
    total_input_size = batch_size * channels * input_size
    
    # Block indices for spatial dimensions
    x_idx = tl.program_id(2) * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_idx = tl.program_id(3) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    
    # Create 2D mask for spatial dimensions
    x_mask = x_idx < width
    y_mask = y_idx < height
    
    # Combine masks
    mask = x_mask & y_mask
    
    # Calculate linear offset within the spatial dimensions
    spatial_offset_first = y_idx * width + x_idx
    
    # Generate random numbers for dropout using batch/channel indices and spatial positions
    # Use deterministic pseudo-random generation for performance
    seeds = batch_idx * 10000 + channel_idx * 100 + spatial_offset_first
    random_vals = tl.rand(seeds)  # Generate random values in [0, 1)
    
    # Calculate dropout mask (1 for keep, 0 for drop)
    dropout_mask = random_vals > dropout_ratio
    
    # Linearized offset in the full tensor
    channel_offset = batch_idx * channels * input_size + channel_idx * input_size
    linear_offsets = channel_offset + spatial_offset_first
    
    # Load input data
    input_data = tl.load(input_ptr + linear_offsets, mask=mask, other=0.0)
    
    # Apply ReLU and Dropout in a single kernel
    relu_output = tl.max(input_data, 0.0)  # ReLU operation
    
    # Apply dropout using scaling
    dropout_ratio_t = tl.literal(dropout_ratio)
    dropout_scale = 1.0 / (1.0 - dropout_ratio_t)
    dropout_output = relu_output * dropout_mask * dropout_scale
    
    # Store both outputs
    tl.store(relu_output_ptr + linear_offsets, relu_output, mask=mask)
    tl.store(dropout_output_ptr + linear_offsets, dropout_output, mask=mask)

@torch.fx.wrap
def fused_relu_dropout_wrapper(in_0):
    # Get tensor dimensions
    batch_size, channels, height, width = in_0.shape
    
    # Prepare output tensors
    relu_output = torch.empty_like(in_0)
    dropout_output = torch.empty_like(in_0)
    
    # Block sizes optimized for 2D operations
    BLOCK_SIZE_X = 16  # Width block size
    BLOCK_SIZE_Y = 16  # Height block size
    
    # Calculate grid dimensions
    grid_z = (batch_size + 63) // 64  # Batch dimension
    grid_c = (channels + 63) // 64    # Channel dimension  
    grid_x = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X  # Width dimension
    grid_y = (height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y  # Height dimension
    
    # Use 4D grid: (batch_groups, channel_groups, grid_x, grid_y)
    grid = (grid_z, grid_c, grid_x, grid_y)
    
    # Launch the fused kernel
    fused_relu_dropout_kernel[grid](
        in_0,
        dropout_output,
        relu_output,
        batch_size,
        channels,
        height,
        width,
        0.1,  # dropout_ratio
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y
    )
    
    return (dropout_output, relu_output)

def replacement_func():
    return fused_relu_dropout_wrapper