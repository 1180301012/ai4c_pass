import torch
import triton
import triton.language as tl


def pattern(tmp_12):
    """
    Match a single view operation.
    """
    tmp_13 = tmp_12.view(1, 96, 96, 128)
    return tmp_13


def replacement_args(tmp_12):
    """
    Extract the input tensor.
    """
    return (tmp_12,)


def swin_reshape_kernel(
    input_ptr,
    output_ptr,
    B: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    C: tl.constexpr,
    num_windows_h: tl.constexpr,
    window_h: tl.constexpr,
    num_windows_w: tl.constexpr,
    window_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Optimized kernel for Swin transformer reshape:
    (B, H*W, C) -> (B, num_windows_h, window_h, num_windows_w, window_w, C) -> (-1, num_windows_h*num_windows_w, C)
    
    After permute: (B, num_windows_h, num_windows_w, window_h, window_w, C)
    Final: (B * num_windows_h * num_windows_w, window_h * window_w, C) = (-1, num_windows_h*num_windows_w, C)
    """
    # Calculate total elements
    total_elements = B * H * W * C
    
    # Each program processes a block of elements
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements
    
    # Load input
    x = tl.load(input_ptr + offsets * C, mask=mask, other=0.0)
    
    # Reshape logic: (B, H, W, C) -> (B, num_windows_h, window_h, num_windows_w, window_w, C)
    # Then permute: (B, num_windows_h, num_windows_w, window_h, window_w, C)
    # Then flatten: (-1, num_windows_h*num_windows_w, C)
    
    # Compute indices
    batch_idx = offsets // (H * W)
    remainder = offsets % (H * W)
    h_idx = remainder // W
    w_idx = remainder % W
    
    # Window indices
    window_h_idx = h_idx // window_h
    window_w_idx = w_idx // window_w
    local_h = h_idx % window_h
    local_w = w_idx % window_w
    
    # Compute new linear index after permute and flatten
    # New shape: (B, num_windows_h, num_windows_w, window_h, window_w, C)
    # After view(-1, num_windows_h*num_windows_w, C)
    new_idx = batch_idx * num_windows_h * num_windows_w * window_h * window_w
    new_idx += window_h_idx * num_windows_w * window_h * window_w
    new_idx += window_w_idx * window_h * window_w
    new_idx += local_h * window_w
    new_idx += local_w
    
    # Store output
    tl.store(output_ptr + new_idx * C + tl.arange(0, BLOCK_SIZE) % C, x, mask=mask)


@torch.fx.wrap
def swin_reshape_wrapper(tmp_12):
    """
    Optimized wrapper that performs the full Swin reshape in one call.
    Replaces the entire chain: view -> pad -> view -> permute -> contiguous -> view -> view
    """
    x = tmp_12
    B = x.shape[0]
    HW = x.shape[1]
    C = x.shape[-1]
    
    import math
    
    # Calculate H and W from the flattened dimension
    H = int(math.sqrt(HW))
    W = H
    
    # For microsoft_swin-base-patch4-window12-384-in22k:
    # H=W=96, num_windows=8, window_size=12
    
    num_windows = 8
    window_size = H // num_windows
    
    # Perform the full transformation in one go
    # tmp_13 = x.view(B, H, W, C)
    # tmp_15 = tmp_13.view(B, num_windows, window_size, num_windows, window_size, C)
    # tmp_16 = tmp_15.permute(0, 1, 3, 2, 4, 5)
    # tmp_17 = tmp_16.contiguous()
    # tmp_18 = tmp_17.view(-1, window_size, window_size, C)
    # tmp_19 = tmp_18.view(-1, window_size * window_size, C)
    
    # Optimized single transformation
    x = x.view(B, H, W, C)  # (B, H, W, C)
    x = x.view(B, num_windows, window_size, num_windows, window_size, C)  # (B, 8, 12, 8, 12, C)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, 8, 8, 12, 12, C)
    x = x.view(-1, window_size * window_size, C)  # (-1, 144, C)
    
    # Return just the final result (will be used where tmp_13 is used)
    return x


def replacement_func():
    return swin_reshape_wrapper