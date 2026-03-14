import torch
import triton
import triton.language as tl

def pattern(x, h, w, c, num_h, window_h, num_w, window_w):
    """
    Pattern: Window partitioning operations
    view -> pad -> view -> permute -> contiguous -> view -> view
    """
    # Reshape to 4D
    x1 = x.view(1, h, w, c)
    # Pad (no-op with all zeros)
    x2 = torch.nn.functional.pad(x1, (0, 0, 0, 0, 0, 0), 'constant', None)
    # Reshape to 6D for window partitioning
    x3 = x2.view(1, num_h, window_h, num_w, window_w, c)
    # Permute to group windows
    x4 = x3.permute(0, 1, 3, 2, 4, 5)
    # Make contiguous
    x5 = x4.contiguous()
    # Flatten windows
    x6 = x5.view(-1, window_h, window_w, c)
    # Final reshape
    x7 = x6.view(-1, window_h * window_w, c)
    return x7

def replacement_args(x, h, w, c, num_h, window_h, num_w, window_w):
    return (x, h, w, c, num_h, window_h, num_w, window_w)

@triton.jit
def window_partition_kernel(
    input_ptr,
    output_ptr,
    seq_len,
    c,
    num_h,
    window_h,
    num_w,
    window_w,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused window partitioning kernel
    Input: [1, seq_len, c] where seq_len = h * w
    Output: [num_windows, window_size, c] where num_windows = num_h * num_w, window_size = window_h * window_w
    """
    pid = tl.program_id(0)
    
    num_windows = num_h * num_w
    window_size = window_h * window_w
    total_elements = num_windows * window_size * c
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Output layout: [num_windows, window_size, c]
    window_idx = offsets // (window_size * c)
    remainder = offsets % (window_size * c)
    pos_in_window = remainder // c
    c_idx = remainder % c
    
    # Decompose window_idx to window coordinates
    win_h = window_idx // num_w
    win_w = window_idx % num_w
    
    # Decompose pos_in_window to position within window
    local_h = pos_in_window // window_w
    local_w = pos_in_window % window_w
    
    # Calculate global position in input
    global_h = win_h * window_h + local_h
    global_w = win_w * window_w + local_w
    
    # Input is [1, h*w, c] = [1, seq_len, c]
    # Linear index: batch=0, seq_pos = global_h * (num_w * window_w) + global_w, channel = c_idx
    h_total = num_h * window_h
    w_total = num_w * window_w
    seq_pos = global_h * w_total + global_w
    input_idx = seq_pos * c + c_idx
    
    # Load and store
    data = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
    tl.store(output_ptr + offsets, data, mask=mask)

@torch.fx.wrap
def fused_window_partition(x, h, w, c, num_h, window_h, num_w, window_w):
    """
    Fused window partitioning
    """
    num_windows = num_h * num_w
    window_size = window_h * window_w
    
    # Output shape: [num_windows, window_size, c]
    output = torch.empty((num_windows, window_size, c), dtype=x.dtype, device=x.device)
    
    total_elements = num_windows * window_size * c
    BLOCK_SIZE = 1024
    grid = ((total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    window_partition_kernel[grid](
        x,
        output,
        h * w,
        c,
        num_h,
        window_h,
        num_w,
        window_w,
        BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_window_partition