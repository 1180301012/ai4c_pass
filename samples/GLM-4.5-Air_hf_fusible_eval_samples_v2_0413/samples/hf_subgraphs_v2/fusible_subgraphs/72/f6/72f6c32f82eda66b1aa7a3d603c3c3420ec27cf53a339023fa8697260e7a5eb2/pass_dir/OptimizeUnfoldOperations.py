import torch
import triton
import triton.language as tl

def pattern(tmp_2):
    tmp_3 = tmp_2.unfold(2, 12, 8)
    tmp_4 = tmp_3.unfold(3, 12, 8)
    return tmp_4

def replacement_args(tmp_2):
    return (tmp_2,)

@triton.jit
def optimized_unfold_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    window_size,
    stride,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Get program ID for parallel execution
    pid = tl.program_id(0)
    
    # Calculate total output elements and distribute work
    total_windows_h = (input_height - window_size) // stride + 1
    total_windows_w = (input_width - window_size) // stride + 1
    total_windows = total_windows_h * total_windows_w
    
    # Map program to specific window
    window_idx = pid
    window_h = window_idx // total_windows_w
    window_w = window_idx % total_windows_w
    
    # Calculate output dimensions after unfolding
    unfolded_height = (input_height - window_size) // stride + 1
    unfolded_width = (input_width - window_size) // stride + 1
    
    # Process each channel in blocks
    for c_base in range(0, channels, BLOCK_SIZE_C):
        for window_c in range(BLOCK_SIZE_C):
            c = c_base + window_c
            if c >= channels:
                break
                
            # Calculate output offset for this batch, channel, and window position
            batch_offset = pid // (channels * unfolded_height * unfolded_width)
            channel_offset = (pid // (unfolded_height * unfolded_width)) % channels
            h_offset = (pid // unfolded_width) % unfolded_height
            w_offset = pid % unfolded_width
            
            # Process window positions
            for win_h in range(window_size):
                for win_w in range(window_size):
                    # Calculate input coordinates
                    in_h = h_offset * stride + win_h
                    in_w = w_offset * stride + win_w
                    
                    # Check bounds and load input value
                    if 0 <= in_h < input_height and 0 <= in_w < input_width:
                        input_idx = (batch_offset * channels + c) * input_height * input_width + in_h * input_width + in_w
                        input_val = tl.load(input_ptr + input_idx)
                    else:
                        input_val = 0.0
                    
                    # Calculate output position in unfolded tensor
                    output_h = h_offset
                    output_w = w_offset
                    output_win_h = win_h
                    output_win_w = win_w
                    
                    # Output dimensions: batch, channels, unfolded_h, window_size, unfolded_w, window_size
                    output_idx = (batch_offset * channels + c) * (unfolded_height * window_size * unfolded_width * window_size) + \
                                output_h * (window_size * unfolded_width * window_size) + \
                                output_win_h * (unfolded_width * window_size) + \
                                output_w * window_size + output_win_w
                    
                    tl.store(output_ptr + output_idx, input_val)

@torch.fx.wrap
def optimized_unfold_2d(tmp_2):
    input_shape = tmp_2.shape
    batch_size = input_shape[0]
    channels = input_shape[1]
    input_height = input_shape[2]
    input_width = input_shape[3]
    
    # Unfolding parameters
    window_size = 12
    stride = 8
    
    # Calculate output dimensions
    unfolded_height = (input_height - window_size) // stride + 1
    unfolded_width = (input_width - window_size) // stride + 1
    
    # Output shape: [batch, channels, unfolded_height, window_size, unfolded_width, window_size]
    output_shape = (batch_size, channels, unfolded_height, window_size, unfolded_width, window_size)
    output_size = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * output_shape[4] * output_shape[5]
    
    output = torch.zeros(output_shape, dtype=tmp_2.dtype, device=tmp_2.device)
    
    # Configure launch parameters
    BLOCK_SIZE_N = 1024  # Number of windows to process simultaneously
    BLOCK_SIZE_C = 32    # Number of channels to process simultaneously
    
    # Calculate grid size
    total_windows = unfolded_height * unfolded_width
    total_elements = batch_size * total_windows * channels
    
    grid = (total_elements + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    
    optimized_unfold_kernel[grid](
        input_ptr=tmp_2,
        output_ptr=output,
        batch_size=batch_size,
        channels=channels,
        input_height=input_height,
        input_width=input_width,
        window_size=window_size,
        stride=stride,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_C=BLOCK_SIZE_C,
    )
    
    # Reshape to match expected format: [batch, channels, unfolded_height, unfolded_width, window_size, window_size]
    return output.view(batch_size, channels, unfolded_height, window_size, unfolded_width, window_size)

def replacement_func():
    return optimized_unfold_2d