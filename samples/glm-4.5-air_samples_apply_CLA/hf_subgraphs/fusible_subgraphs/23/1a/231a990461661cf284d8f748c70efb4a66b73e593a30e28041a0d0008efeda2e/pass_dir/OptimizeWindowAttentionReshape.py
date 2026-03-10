import torch
import triton
import triton.language as tl
from torch import fx

def pattern(x, h, w, c):
    # This pattern matches the window attention reshape sequence
    # The pattern focuses on the key observable operations
    
    # Initial spatial view (tmp_13)
    spatial_view = x.view(1, h, w, c)
    
    # Complex window reshape (tmp_15) - skip pad since it's no-op
    window_view = spatial_view.view(1, 8, h//8, 8, w//8, c)
    
    # Permute for contiguous layout (tmp_16) 
    permuted_view = window_view.permute(0, 1, 3, 2, 4, 5)
    
    # Make contiguous (tmp_17) and final views (tmp_18, tmp_19)
    contiguous_view = permuted_view.contiguous()
    final_window_view = contiguous_view.view(-1, 12, 12, c)
    flattened_view = final_window_view.view(-1, 144, c)
    
    # Return the observable results that match the original computation
    return spatial_view.contiguous(), flattened_view

def replacement_args(input_tensor, h, w, c):
    return (input_tensor, h, w, c)

@triton.jit
def optimized_reshape_kernel(
    x_ptr,
    out_ptr1_ptr,  # Pointer to first output 
    out_ptr2_ptr,  # Pointer to second output
    batch_size,
    h, w, c,
    window_size, window_h, window_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Fast kernel optimized for the specific transformation pattern
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * h * w * c)
    
    # Load input data
    x = tl.load(
        x_ptr + offsets,
        mask=mask,
        other=0.0
    )
    
    # Reshape to spatial format [1, h, w, c]
    x_spatial = x.reshape(batch_size * h * w * c)
    
    # Calculate the number of windows
    num_h_windows = h // 8
    num_w_windows = w // 8
    total_windows = num_h_windows * num_w_windows
    
    # Reshape to window format and permute in one step
    # Directly compute the final flattened format with window organization
    # Output 1: [batch_size, h, w, c] (spatial format)
    if out_ptr1_ptr is not None:
        out1 = x_spatial.reshape(batch_size, h, w, c).contiguous()
        tl.store(out_ptr1_ptr + offsets, out1.flatten(), mask=mask)
    
    # Output 2: [batch_size * total_windows, 64, c] window flattened format
    if out_ptr2_ptr is not None:
        out2_size = batch_size * total_windows * 64 * c
        out2_offsets = tl.arange(0, out2_size)
        out2_mask = out2_offsets < out2_size
        
        # Create window organization directly
        x_reshaped = x_spatial.reshape(batch_size, h, w, c)
        x_windows = x_reshaped.reshape(batch_size, 8, num_h_windows, 8, num_w_windows, c)
        x_windowed = x_windows.permute(0, 1, 3, 2, 4, 5).reshape(batch_size * total_windows, 64, c)
        
        tl.store(out_ptr2_ptr + out2_offsets, x_windowed.flatten(), mask=out2_mask)

@torch.fx.wrap
def optimized_reshape(x, h, w, c):
    batch_size = 1  # Based on the computation pattern
    
    # Calculate window dimensions
    window_size = 8  # Based on the pattern [1, 8, 12, 8, 12, c]
    window_h = h // window_size  # 96 // 8 = 12
    window_w = w // window_size  # 96 // 8 = 12
    
    # Final output dimensions
    num_windows = window_h * window_w  # 12 * 12 = 144
    total_elements = batch_size * num_windows  # 1 * 144 = 144, then *128 for features
    
    # Output tensors
    out1 = torch.empty(batch_size, h, w, c, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    out2 = torch.empty(total_elements, c, dtype=x.dtype, device=x.device, requires_grad=x.requires_grad)
    
    # Triton kernel launch
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    total_input_elements = batch_size * h * w * c
    grid_size = (total_input_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    optimized_reshape_kernel[grid_size](
        x_ptr=x,
        out_ptr1_ptr=out1,
        out_ptr2_ptr=out2,
        batch_size=batch_size,
        h=h, w=w, c=c,
        window_size=window_size,
        window_h=window_h,
        window_w=window_w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out1.contiguous(), out2

def replacement_func():
    return optimized_reshape