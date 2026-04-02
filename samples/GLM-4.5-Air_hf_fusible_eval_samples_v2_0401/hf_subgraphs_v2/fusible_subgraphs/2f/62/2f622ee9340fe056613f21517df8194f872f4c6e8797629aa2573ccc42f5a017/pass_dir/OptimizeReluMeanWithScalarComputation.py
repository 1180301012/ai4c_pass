import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    """
    Pattern matching the computation:
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // div_constant
    tmp_2 = torch.sym_sum([1, tmp_1])
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)
    """
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_1 = in_0 // 8  # This will be matched dynamically
    tmp_2 = torch.sym_sum([1, tmp_1])
    tmp_3 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_3)

def replacement_args(in_0, in_1):
    # Extract inputs for the replacement
    return (in_0, in_1)

@triton.jit
def compute_division(x_val):
    """Compute x // 8 for the scalar operation"""
    return x_val // 8

@triton.jit
def relu_mean_kernel(
    input_ptr,
    mean_out_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_Z: tl.constexpr
):
    # Calculate grid position
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    
    # Create offset tensors
    x_offsets = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    y_offsets = pid_y * BLOCK_SIZE_Y * width + tl.arange(0, BLOCK_SIZE_Y)
    z_offsets = pid_z * BLOCK_SIZE_Z + tl.arange(0, BLOCK_SIZE_Z)
    
    # Create final pointer offsets
    offsets = (z_offsets[:, None, None] * height * width + 
               y_offsets[None, :, None] * width + 
               x_offsets[None, None, :])
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=offsets < n_channels * height * width, other=0.0)
    
    # Apply ReLU operation
    relu_result = tl.maximum(input_data, 0.0)
    
    # Store ReLU result to input_ptr (inplace operation)
    tl.store(input_ptr + offsets, relu_result, mask=offsets < n_channels * height * width)
    
    # Compute mean - we need to sum first then divide
    # For this kernel, we'll compute per-channel mean
    pid_y = tl.program_id(1)  # Channel index
    channel_start = (pid_y * BLOCK_SIZE_Y) * height * width
    channel_end = ((pid_y + 1) * BLOCK_SIZE_Y) * height * width
    
    # Sum all elements in the channel
    channel_sum = 0.0
    for i in range(height):
        for j in range(width):
            offset = channel_start + i * width + j
            if offset < n_channels * height * width:
                channel_sum += relu_result[pid_y * height * width + i * width + j]
    
    # Compute mean
    mean_val = channel_sum / (height * width)
    
    # Store mean result
    tl.store(mean_out_ptr + pid_y, mean_val)

@torch.fx.wrap
def optimized_replacement(in_0, in_1):
    # Handle the scalar computation first
    scalar_input = in_0
    if hasattr(scalar_input, 'numel') and scalar_input.numel() == 1:
        scalar_val = scalar_input.item()
    else:
        scalar_val = scalar_input
    
    # Compute 1 + (scalar // 8) (or other divisor)
    # We need to determine the divisor dynamically based on pattern matching
    # For now, let's use a generic approach
    divisor = 8  # This needs to be determined dynamically
    
    # Get input tensor shape
    input_shape = in_1.shape
    n_channels, height, width = input_shape[1], input_shape[2], input_shape[3]
    
    # Create output tensor for mean result
    mean_result = torch.empty((n_channels,), dtype=in_1.dtype, device=in_1.device)
    
    # Launch Triton kernel for fused ReLU and mean computation
    total_elements = n_channels * height * width
    
    # Choose block sizes based on tensor dimensions
    BLOCK_SIZE_X = min(64, width)
    BLOCK_SIZE_Y = min(64, n_channels)
    BLOCK_SIZE_Z = 1
    
    grid_x = (width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_y = (n_channels + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_z = BLOCK_SIZE_Z
    
    # Launch kernel
    relu_mean_kernel[(grid_x, grid_y, grid_z)](
        in_1,
        mean_result,
        n_channels,
        height,
        width,
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_Z
    )
    
    # Reshape mean result to match original output shape
    mean_result_reshaped = mean_result.view(1, n_channels, 1, 1)
    
    # Compute scalar result
    scalar_result = 1 + (scalar_val // divisor)
    
    # Return original ReLU result (inplace) and computed mean
    return (in_1, mean_result_reshaped.to(in_1.dtype))

def replacement_func():
    return optimized_replacement