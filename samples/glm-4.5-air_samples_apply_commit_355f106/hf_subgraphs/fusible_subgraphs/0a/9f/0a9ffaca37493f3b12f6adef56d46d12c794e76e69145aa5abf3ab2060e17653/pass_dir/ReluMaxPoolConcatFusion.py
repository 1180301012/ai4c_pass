import torch
import triton
import triton.language as tl
from torch import Tensor

# Pattern matching function - matches the entire computation chain
def pattern(input_tensor):
    # ReLU activation (inplace)
    tmp_0 = torch.nn.functional.relu(input_tensor, inplace=True)
    # Three identical max_pool2d operations
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    # Concatenation along dimension 1
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

# Argument extraction function
def replacement_args(input_tensor):
    return (input_tensor,)

# Helper function for ReLU
@triton.jit
def relu_kernel(
    input_ptr,
    output_ptr,
    n_elements: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * 1024 + tl.arange(0, 1024)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.maximum(x, 0.0)
    tl.store(output_ptr + offsets, y, mask=mask)

# Optimized Triton kernel for ReLU + MaxPool2d + Concatenation
@triton.jit
def fusion_relu_maxpool_concat_kernel(
    input_ptr,
    relu_output_ptr,
    pool_output_ptr,
    batch_size,
    input_channels,
    input_height,
    input_width,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    # Get program IDs
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    pid_b = tl.program_id(2)
    
    # Compute output coordinates for max_pool
    output_y = pid_y * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    output_x = pid_x * BLOCK_SIZE_X + tl.arange(0, BLOCK_SIZE_X)
    
    # Create 2D coordinate grid for the 5x5 window with padding 2
    window_y = output_y.unsqueeze(1) + tl.arange(-2, 3)
    window_x = output_x.unsqueeze(0) + tl.arange(-2, 3)
    
    # Flatten coordinates
    window_y_flat = window_y.reshape(-1)
    window_x_flat = window_x.reshape(-1)
    
    # Process ReLU and MaxPool for one channel
    if pid_b < batch_size * input_channels:
        # Calculate channel and position indices
        channel = pid_b // input_height // input_width
        pos_y = (pid_b // input_width) % input_height
        pos_x = pid_b % input_width
        
        # For ReLU: load and apply activation
        relu_base = relu_output_ptr + pid_b
        input_val = tl.load(input_ptr + relu_base)
        relu_val = tl.maximum(input_val, 0.0)
        tl.store(relu_base, relu_val)
        
        # For MaxPool: load neighborhood and compute max
        input_ptr_base = input_ptr + pid_b * input_height * input_width
        input_values = tl.load(
            input_ptr_base + 
            ((window_y_flat * input_width + window_x_flat) + (pos_y - 2) * input_width + (pos_x - 2)).to(tl.int64),
            mask=((window_y_flat * input_width + window_x_flat) + (pos_y - 2) * input_width + (pos_x - 2)) < (input_height * input_width),
            other=-float('inf')
        )
        
        # Compute max over neighborhood
        max_val = tl.max(input_values)
        
        # Store max_pool result
        pool_output_base = pool_output_ptr + pid_b
        tl.store(pool_output_base, max_val)

@torch.fx.wrap
def optimized_fusion(input_tensor):
    # Get input dimensions
    batch_size, input_channels, input_height, input_width = input_tensor.shape
    
    # Create output tensors
    relu_output = torch.empty_like(input_tensor)
    pool_output = torch.empty_like(input_tensor)
    concat_output = torch.empty(batch_size, input_channels * 4, input_height, input_width, 
                                dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch ReLU kernel
    n_elements = input_tensor.numel()
    grid = (n_elements + 1023) // 1024
    relu_kernel[grid](input_tensor, relu_output, n_elements)
    
    # Launch optimized max_pool2d kernel
    BLOCK_SIZE_Y = 16 if input_height >= 16 else input_height
    BLOCK_SIZE_X = 16 if input_width >= 16 else input_width
    
    grid_y = (input_height + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
    grid_x = (input_width + BLOCK_SIZE_X - 1) // BLOCK_SIZE_X
    grid_b = batch_size * input_channels
    
    fusion_relu_maxpool_concat_kernel[(grid_y, grid_x, grid_b)](
        input_tensor,
        relu_output,
        pool_output,
        batch_size,
        input_channels,
        input_height,
        input_width,
        BLOCK_SIZE_Y,
        BLOCK_SIZE_X,
    )
    
    # Perform concatenation on GPU
    # Create three references to the same max_pool result
    pool_repeated = pool_output
    # Concatenate along dimension 1
    concat_output = torch.cat([relu_output, pool_repeated, pool_repeated, pool_repeated], dim=1)
    
    return concat_output

# Replacement function
def replacement_func():
    return optimized_fusion