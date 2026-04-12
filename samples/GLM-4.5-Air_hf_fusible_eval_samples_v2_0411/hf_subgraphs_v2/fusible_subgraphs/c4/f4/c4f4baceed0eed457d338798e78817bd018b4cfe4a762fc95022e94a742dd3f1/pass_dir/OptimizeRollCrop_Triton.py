import torch
import triton
import triton.language as tl

def pattern(x, view_shape, roll_shifts, slice_dims):
    """
    Pattern matches: contiguous → view → torch.roll → crop operations
    This matches the exact computation structure from model.py
    """
    tmp_2 = x.contiguous()
    tmp_3 = tmp_2.view(view_shape)
    tmp_4 = torch.roll(tmp_3, shifts=roll_shifts, dims=(1, 2))
    tmp_5 = tmp_4[(slice(None, None, None), slice(None, slice_dims[0], None), slice(None, slice_dims[1], None), slice(None, None, None))]
    return tmp_5

def replacement_args(x, view_shape, roll_shifts, slice_dims):
    return (x, view_shape, roll_shifts, slice_dims)

@triton.jit
def roll_crop_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    input_height,
    input_width, 
    input_channels,
    roll_shift_h,
    roll_shift_w,
    crop_height,
    crop_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
    BLOCK_SIZE_C: tl.constexpr,
):
    # Calculate grid coordinates
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    pid_c = tl.program_id(2)
    
    # Calculate spatial offsets
    h_offset = pid_h * BLOCK_SIZE_H
    w_offset = pid_w * BLOCK_SIZE_W
    c_offset = pid_c * BLOCK_SIZE_C
    
    # Calculate output coordinates
    h_out = h_offset + tl.arange(0, BLOCK_SIZE_H)
    w_out = w_offset + tl.arange(0, BLOCK_SIZE_W)
    c_out = c_offset + tl.arange(0, BLOCK_SIZE_C)
    
    # Create masks
    h_mask = h_out < crop_height
    w_mask = w_out < crop_width
    c_mask = c_out < input_channels
    
    # For each output position, calculate corresponding input position with roll
    for h, h_valid in tl.static_if(h_mask)(h_out, h_mask):
        for w, w_valid in tl.static_if(w_mask)(w_out, w_mask):
            for c, c_valid in tl.static_if(c_mask)(c_out, c_mask):
                # Apply roll transformation in reverse
                h_in = (h - roll_shift_h) % input_height
                w_in = (w - roll_shift_w) % input_width
                
                # Calculate linear index
                input_idx = h_in * input_width * input_channels + w_in * input_channels + c
                output_idx = h * crop_width * input_channels + w * input_channels + c
                
                # Load input and store output
                val = tl.load(input_ptr + input_idx, mask=(h_valid & w_valid & c_valid), other=0.0)
                tl.store(output_ptr + output_idx, val, mask=(h_valid & w_valid & c_valid))

@torch.fx.wrap
def optimized_roll_crop(x, view_shape, roll_shifts, slice_dims):
    # Extract dimensions
    batch_size, height, width, channels = view_shape
    
    if len(x.shape) == 6:  # Original shape is [1, D1, H1, W1, C] 
        # Determine actual spatial dimensions
        height = x.shape[2]  # H1
        width = x.shape[3]   # W1
        channels = x.shape[5]  # C
        batch_size = x.shape[0] * x.shape[1] * x.shape[4]  # Combined dimensions
    
    # Crop dimensions
    crop_h, crop_w = slice_dims
    
    # Create output tensor
    output_shape = (batch_size, crop_h, crop_w, channels)
    output = torch.empty(output_shape, dtype=x.dtype, device=x.device)
    
    # Calculate block sizes
    BLOCK_SIZE_H = 32
    BLOCK_SIZE_W = 32
    BLOCK_SIZE_C = min(32, channels)
    
    # Calculate grid size
    grid_h = (crop_h + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H
    grid_w = (crop_w + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W
    grid_c = (channels + BLOCK_SIZE_C - 1) // BLOCK_SIZE_C
    
    total_elements = batch_size * crop_h * crop_w * channels
    x_contiguous = x.contiguous()
    
    # Launch kernel
    roll_crop_kernel[(grid_h, grid_w, grid_c, batch_size)](
        x_contiguous,
        output,
        total_elements,
        height,
        width,
        channels,
        roll_shifts[0],
        roll_shifts[1],
        crop_h,
        crop_w,
        BLOCK_SIZE_H,
        BLOCK_SIZE_W,
        BLOCK_SIZE_C,
    )
    
    return output

def replacement_func():
    return optimized_roll_crop