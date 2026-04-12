import torch
import triton
import triton.language as tl

def pattern(tmp_0):
    tmp_1 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_2 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_3 = torch.nn.functional.max_pool2d(tmp_0, 5, 1, 2, 1, ceil_mode=False, return_indices=False)
    tmp_4 = torch.cat([tmp_0, tmp_1, tmp_2, tmp_3], 1)
    return tmp_4

def replacement_args(tmp_0):
    return (tmp_0,)

@triton.jit
def max_pool_2d_kernel(
    input_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_x: tl.constexpr,
    BLOCK_SIZE_y: tl.constexpr,
):
    """Standard max_pool2d implementation with 5x5 kernel and 2 pixel padding"""
    pid_height = tl.program_id(0)
    pid_width = tl.program_id(1)
    pid_batch = tl.program_id(2)
    
    # Each program handles one spatial position in the output
    y = pid_height
    x = pid_width
    
    # Only compute within output bounds
    if y >= height or x >= width:
        return
    
    # Compute max over 5x5 window with 2 pixel padding on each side
    max_val = -float('inf')
    
    # Iterate over 5x5 window with effective padding of 2 pixels
    for dy in range(-2, 3):  # -2, -1, 0, 1, 2
        for dx in range(-2, 3):  # -2, -1, 0, 1, 2
            # Calculate input position with padding
            in_y = y + dy  # padding handled by clamping to valid range
            in_x = x + dx
            
            # Only access valid positions
            if 0 <= in_y < height and 0 <= in_x < width:
                for c in range(n_channels):
                    # Calculate linear index
                    idx = ((pid_batch * n_channels + c) * height + in_y) * width + in_x
                    val = tl.load(input_ptr + idx)
                    max_val = tl.maximum(max_val, val)
    
    # Store result
    output_idx = ((pid_batch * n_channels + 0) * height + y) * width + x
    tl.store(output_ptr, max_val)

@triton.jit
def fused_kernel(
    input_ptr,
    output1_ptr, 
    output2_ptr,
    output3_ptr,
    n_channels,
    height,
    width,
    BLOCK_SIZE_M: tl.constexpr,
):
    """Kernel that computes three identical max_pool2d operations with proper indexing"""
    pid = tl.program_id(0)
    
    # Each program handles one spatial position across all channels
    x = pid % width
    y = (pid // width) % height
    batch = pid // (width * height)
    
    # Only compute within bounds
    if x >= width or y >= height or batch >= 1:
        return
    
    # We'll handle all channels at once in a loop
    for c in range(n_channels):
        max_val1 = -float('inf')
        max_val2 = -float('inf') 
        max_val3 = -float('inf')
        
        # Compute max over 5x5 window with 2 pixel padding
        for dy in range(-2, 3):  # -2 to +2
            for dx in range(-2, 3):  # -2 to +2
                in_y = y + dy
                in_x = x + dx
                
                # Only valid spatial positions
                if 0 <= in_y < height and 0 <= in_x < width:
                    # Calculate pointer offset
                    offset = ((batch * n_channels + c) * height + in_y) * width + in_x
                    val = tl.load(input_ptr + offset)
                    
                    # All three max pools compute the same result
                    max_val1 = tl.maximum(max_val1, val)
                    max_val2 = tl.maximum(max_val2, val)
                    max_val3 = tl.maximum(max_val3, val)
        
        # Store results for all three outputs, same max value
        out_offset1 = ((batch * n_channels + c) * height + y) * width + x
        out_offset2 = ((batch * n_channels + c) * height + y) * width + x
        out_offset3 = ((batch * n_channels + c) * height + y) * width + x
        
        tl.store(output1_ptr + out_offset1, max_val1)
        tl.store(output2_ptr + out_offset2, max_val2)
        tl.store(output3_ptr + out_offset3, max_val3)

@torch.fx.wrap  
def fused_max_pool_triton(x):
    batch_size, channels, height, width = x.shape
    
    # Allocate output tensors for the three max pools  
    x_pool1 = torch.empty((batch_size, channels, height, width), dtype=x.dtype, device=x.device)
    x_pool2 = torch.empty((batch_size, channels, height, width), dtype=x.dtype, device=x.device)
    x_pool3 = torch.empty((batch_size, channels, height, width), dtype=x.dtype, device=x.device)
    
    # Total number of spatial positions to process
    total_positions = height * width
    
    # Launch fused kernel with 1D grid over spatial positions
    grid = (total_positions,)
    
    fused_kernel[grid](
        x,
        x_pool1,
        x_pool2,
        x_pool3,
        channels,
        height, 
        width,
        1  # BLOCK_SIZE_M parameter (not used in this simplified version)
    )
    
    # Concatenate original + 3 max pools along channel dimension  
    result = torch.cat([x, x_pool1, x_pool2, x_pool3], dim=1)
    return result

def replacement_func():
    return fused_max_pool_triton