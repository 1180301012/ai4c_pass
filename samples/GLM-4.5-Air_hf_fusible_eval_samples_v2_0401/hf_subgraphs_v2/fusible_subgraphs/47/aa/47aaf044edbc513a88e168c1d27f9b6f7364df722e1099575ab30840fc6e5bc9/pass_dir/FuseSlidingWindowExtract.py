import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern matches the sliding window extraction sequence:
    contiguous -> unsqueeze(-1) -> unfold with [9,1] kernel -> transpose(1,2) -> reshape -> final reshape
    """
    tmp_0 = x.contiguous()
    tmp_1 = tmp_0.unsqueeze(-1)
    tmp_2 = torch.nn.functional.unfold(tmp_1, kernel_size=[9, 1], dilation=1, padding=[4, 0], stride=1)
    tmp_3 = tmp_2.transpose(1, 2)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5

def replacement_args(x):
    return (x,)

@triton.jit
def sliding_window_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    input_channels,
    input_height,
    output_channels_per_group,
    window_size,
    num_groups,
    BLOCK_SIZE_Y: tl.constexpr,
    BLOCK_SIZE_X: tl.constexpr,
):
    """
    Custom Triton kernel for sliding window extraction.
    Extracts 9x1 sliding windows and reshapes output efficiently.
    """
    pid = tl.program_id(0)
    
    # Determine which group and position this program handles
    group_id = pid // input_height
    position_in_group = pid % input_height
    
    if group_id >= num_groups or position_in_group >= input_height:
        return
    
    # Calculate output indices
    batch_idx = 0  # Single batch
    window_idx = position_in_group
    group_output_idx = group_id
    
    # Calculate starting position for the window in input
    window_start_y = position_in_group
    
    # Process each output channel in this group
    for out_ch_idx in range(output_channels_per_group):
        out_base = (out_ptr + 
                   ((batch_idx * num_groups * input_height * output_channels_per_group + 
                     group_id * input_height * output_channels_per_group + 
                     position_in_group * output_channels_per_group + 
                     out_ch_idx) * window_size))
        
        # Extract 9-element window from input
        window_offset = (group_output_idx * input_channels * input_height + 
                        out_ch_idx * input_height + 
                        window_start_y)
        
        # Load the 9 elements from the sliding window
        for i in range(window_size):
            input_offset = window_offset + i
            if input_offset < batch_size * input_channels * input_height:
                value = tl.load(x_ptr + input_offset, mask=input_offset < batch_size * input_channels * input_height, other=0.0)
                tl.store(out_base + i, value)

@torch.fx.wrap
def optimized_sliding_window_extract(x, num_groups=8, output_channels_per_group=2, window_size=9):
    """
    Optimized sliding window extraction function that fuses multiple operations.
    """
    batch_size, input_channels, input_height = x.shape
    total_groups = (input_channels + num_groups - 1) // num_groups
    
    # Ensure input is contiguous
    if not x.is_contiguous():
        x = x.contiguous()
    
    # Calculate output shape: [batch_size * total_groups * input_height, output_channels_per_group, window_size]
    # But the original pattern produces [-1, 8, 9] = [batch_size * total_groups * input_height, output_channels_per_group, window_size]
    output_shape = (batch_size * total_groups * input_height, output_channels_per_group, window_size)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Set up launch grid
    total_elements = batch_size * total_groups * input_height
    BLOCK_SIZE_y = 32  # Number of positions to process
    BLOCK_SIZE_x = 1   # Number of groups to process
    
    # Calculate grid dimensions
    grid_y = (total_elements + BLOCK_SIZE_y - 1) // BLOCK_SIZE_y
    grid_x = 1
    grid_size = grid_y * grid_x
    
    # Launch kernel
    sliding_window_kernel[grid_size, (
        BLOCK_SIZE_y,
        BLOCK_SIZE_x
    )](
        x_ptr=x,
        out_ptr=output,
        batch_size=batch_size,
        input_channels=input_channels,
        input_height=input_height,
        output_channels_per_group=output_channels_per_group,
        window_size=window_size,
        num_groups=total_groups
    )
    
    return output

def replacement_func():
    return optimized_sliding_window_extract