import torch
import triton
import triton.language as tl
from torch import Tensor

# Pattern matching function - matches the redundant max_pool2d computation
def pattern(input_tensor):
    # First max_pool2d operation
    tmp_1 = torch.nn.functional.max_pool2d(input_tensor, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    # Second max_pool2d operation (recomputation)
    tmp_2 = torch.nn.functional.max_pool2d(input_tensor, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    # Third max_pool2d operation (recomputation)
    tmp_3 = torch.nn.functional.max_pool2d(input_tensor, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    return tmp_1, tmp_2, tmp_3

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Optimized Triton kernel for max_pool2d with padding=2, stride=1, kernel_size=5
@triton.jit
def max_pool2d_optimized_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    input_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    # Program identifiers
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute output coordinates
    output_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    output_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    # Create 2D coordinate grid for the window
    window_y = output_y.unsqueeze(1) + tl.arange(-2, 3)  # 5x5 window with padding 2
    window_x = output_x.unsqueeze(0) + tl.arange(-2, 3)
    
    # Flatten coordinates for vectorized loading
    window_y_flat = window_y.reshape(-1)
    window_x_flat = window_x.reshape(-1)
    
    # Load input elements with bounds checking
    input_ptr_base = input_ptr + pid_b * input_channels * input_height * input_width
    
    # Load all window elements
    input_values = tl.load(
        input_ptr_base + 
        (window_y_flat[:, None] * input_width + window_x_flat[None, :]).flatten(),
        mask=(window_y_flat[:, None] * input_width + window_x_flat[None, :]) < (input_channels * input_height * input_width),
        other=-float('inf')
    )
    
    # Reshape back to window shape and compute max
    window_values = input_values.reshape(5, 5)
    max_val = tl.max(window_values)
    
    # Store the result
    output_idx = pid_b * output_channels * output_height * output_width + pid_y * output_width + pid_x
    tl.store(output_ptr + output_idx, max_val)

@torch.fx.wrap
def optimized_max_pool2d_once(input_tensor):
    # Get input dimensions
    batch_size, input_channels, input_height, input_width = input_tensor.shape
    
    # For max_pool2d with kernel_size=5, stride=1, padding=2
    output_height = input_height
    output_width = input_width
    
    # Optimized block sizes
    BLOCK_SIZE_Y = 16 if output_height >= 16 else output_height
    BLOCK_SIZE_X = 16 if output_width >= 16 else output_width
    
    # Calculate grid size
    grid_y = (output_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (output_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_b = batch_size * input_channels  # Process each channel separately
    
    # Create output tensor
    output_shape = (batch_size, input_channels, output_height, output_width)
    result = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel for max_pool2d computation
    max_pool2d_optimized_kernel[(grid_y, grid_x, grid_b)](
        input_tensor,
        result,
        batch_size,
        input_channels,
        input_height,
        input_width,
        output_height,
        output_width,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_X,
    )
    
    return result

# Replacement function
def replacement_func():
    def optimized_forward(input_tensor):
        # Compute max_pool2d once
        max_pool_result = optimized_max_pool2d_once(input_tensor)
        # Return three references to the same result
        return max_pool_result, max_pool_result, max_pool_result
    
    return optimized_forward