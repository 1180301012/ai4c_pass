import torch
import triton
import triton.language as tl

@triton.jit
def window_partition_kernel(
    x_ptr,           # Input [1, H, W, C]
    out_ptr,         # Output [1, num_h_windows, num_w_windows, window_size, window_size, C]
    height, width, channels,
    window_size_h, window_size_w,
    BLOCK_SIZE_WINDOW_H: tl.constexpr,
    BLOCK_SIZE_WINDOW_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr
):
    # Program ID for window position
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate window boundaries
    start_h = pid_h * window_size_h
    start_w = pid_w * window_size_w
    
    # If out of bounds, return
    if start_h >= height or start_w >= width:
        return
    
    # Calculate actual window size (might be smaller at edges)
    win_h = min(window_size_h, height - start_h)
    win_w = min(window_size_w, width - start_w)
    
    # Load single elements from the window
    offsets_h = tl.arange(0, win_h)
    offsets_w = tl.arange(0, win_w)
    offsets_c = pid_c * BLOCK_SIZE_C + tl.arange(0, BLOCK_SIZE_C)
    
    # Create coordinate grid
    h_coords = start_h + offsets_h[:, None]
    w_coords = start_w + offsets_w[None, :]
    
    # Load input window
    x_base = h_coords * width + w_coords
    x_offsets = x_base * channels + offsets_c[None, None, :]
    
    # Create output dimensions
    out_offsets = (pid_h * window_size_w + pid_w) * BLOCK_SIZE_C + offsets_c
    
    # Output layout: [start_h, start_w, c, window_h, window_w]
    for i, h in enumerate(offsets_h[:win_h]):
        for j, w in enumerate(offsets_w[:win_w]):
            x_val = tl.load(x_ptr + x_offsets[i, j, :], mask=offsets_c < channels, other=0.0)
            tl.store(out_ptr + out_offsets, x_val, mask=offsets_c < channels)

@torch.fx.wrap
def optimized_window_partition(x, window_size_h=7, window_size_w=7):
    """
    Optimized window partitioning that directly converts [1, H, W, C] to 
    [1, num_h_windows, num_w_windows, window_size, window_size, C] in one step
    """
    batch, height, width, channels = x.shape
    
    # Calculate number of windows
    num_h_windows = (height + window_size_h - 1) // window_size_h
    num_w_windows = (width + window_size_w - 1) // window_size_w
    
    # Create output tensor directly in the final format
    output = torch.empty(1, num_h_windows, num_w_windows, window_size_h, window_size_w, channels,
                        dtype=x.dtype, device=x.device)
    
    # Triton kernel parameters
    BLOCK_SIZE_WINDOW_H = min(window_size_h, 8)
    BLOCK_SIZE_WINDOW_W = min(window_size_w, 8)
    BLOCK_SIZE_C = min(128, 32)
    
    # Launch grid
    grid = (num_h_windows, num_w_windows, (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C)
    
    window_partition_kernel[grid](
        x_ptr=x,
        out_ptr=output,
        height=height, width=width, channels=channels,
        window_size_h=window_size_h, window_size_w=window_size_w,
        BLOCK_SIZE_WINDOW_H=BLOCK_SIZE_WINDOW_H,
        BLOCK_SIZE_WINDOW_W=BLOCK_SIZE_WINDOW_W,
        BLOCK_SIZE_C=BLOCK_SIZE_C
    )
    
    return output

def pattern(x, pad, mode, value):
    tmp_13 = x.view(1, x.shape[1], x.shape[2], x.shape[3])
    tmp_14 = torch.nn.functional.pad(tmp_13, pad, mode, value)
    tmp_15 = tmp_14.view(1, x.shape[1]//7, 7, x.shape[2]//7, 7, x.shape[3])
    tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
    tmp_17 = tmp_16.contiguous()
    tmp_18 = tmp_17.view(-1, 7, 7, x.shape[3])
    tmp_19 = tmp_18.view(-1, 49, x.shape[3])
    return tmp_19

def replacement_args(x, pad, mode, value):
    return (x, pad, mode, value)

def replacement_func():
    return optimized_window_partition