import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for interpolation + concatenation + stacking computation
    """
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.nn.functional.interpolate(in_0, size=(40, 40), mode='nearest')
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(40, 40), mode='nearest')
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return tmp_3

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def nearest_interpolate_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    input_c,
    input_h,
    input_w,
    output_h,
    output_w,
    output_c,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    
    # Calculate scaling factors
    y_scale = input_h / output_h
    x_scale = input_w / output_w
    
    # Output coordinates
    y_out = tl.program_id(2) * BLOCK_SIZE_Y + tl.arange(0, BLOCK_SIZE_Y)
    x_out = tl.range(0, BLOCK_SIZE_X)
    
    # Scale to input coordinates (nearest neighbor)
    y_in = (y_out * y_scale).to(tl.int32)
    x_in = (x_out * x_scale).to(tl.int32)
    
    # Clamp to valid input range
    y_in = tl.minimum(y_in, input_h - 1)
    x_in = tl.minimum(x_in, input_w - 1)
    
    # Address calculation
    input_offset = batch_idx * input_c * input_h * input_w + channel_idx * input_h * input_w + y_in * input_w + x_in
    output_offset = batch_idx * output_c * output_h * output_w + channel_idx * output_h * output_w + y_out * output_w + x_out
    
    # Load input and store output
    x = tl.load(x_ptr + input_offset, mask=(y_out < output_h)[:, None] & (x_out[None, :] < output_w), other=0.0)
    tl.store(out_ptr + output_offset, x, mask=(y_out < output_h)[:, None] & (x_out[None, :] < output_w))

@triton.jit
def concat_kernel(
    ptr1,
    ptr2,
    out_ptr,
    batch_size,
    c1,
    c2,
    h,
    w,
    BLOCK_SIZE_X: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    channel_idx = tl.program_id(1)
    y_idx = tl.program_id(2)
    
    # Within this block, calculate x coordinates
    x_idx = tl.arange(0, BLOCK_SIZE_X)
    
    # Determine if we're in first or second tensor based on channel index
    if channel_idx < c1:
        # First tensor
        src_ptr = ptr1
        src_offset = batch_idx * c1 * h * w + channel_idx * h * w + y_idx * w + x_idx
    else:
        # Second tensor
        src_ptr = ptr2
        src_channel_idx = channel_idx - c1
        src_offset = batch_idx * c2 * h * w + src_channel_idx * h * w + y_idx * w + x_idx
    
    # Destination offset
    c_total = c1 + c2
    dst_offset = batch_idx * c_total * h * w + channel_idx * h * w + y_idx * w + x_idx
    
    # Load and store with bounds checking
    mask = x_idx < w
    if y_idx < h:
        x = tl.load(src_ptr + src_offset, mask=mask, other=0.0)
        tl.store(out_ptr + dst_offset, x, mask=mask)

@torch.fx.wrap
def optimized_interpolate_upstack(in_0, in_1, in_2, in_3):
    """
    Optimized implementation that fuses interpolation, concatenation, and stacking
    """
    batch_size_0, c_0, h_0, w_0 = in_0.shape
    batch_size_1, c_1, h_1, w_1 = in_1.shape
    batch_size_2, c_2, h_2, w_2 = in_2.shape
    batch_size_3, c_3, h_3, w_3 = in_3.shape
    
    # Final output shape after operations
    output_channels = c_2 + c_3  # Concat result channels
    final_channels = c_0 + c_1 + output_channels  # Stack result channels
    
    # Output tensor
    output = torch.empty((batch_size_0, final_channels, 40, 40), dtype=in_0.dtype, device=in_0.device)
    
    # Block sizes for Triton kernel
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    
    # Kernel 1: Interpolate in_0 -> output[:, 0:c_0, :, :]
    if h_0 != 40 or w_0 != 40:
        grid_x = (batch_size_0 + 7) // 8
        grid_y = (c_0 + 7) // 8
        grid_z = (40 + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
        
        nearest_interpolate_kernel[(grid_x, grid_y, grid_z)](
            in_0,
            output[:, 0:c_0, :, :],
            batch_size_0,
            c_0,
            h_0,
            w_0,
            40,
            40,
            c_0,
            BLOCK_SIZE_X,
            BLOCK_SIZE_Y,
        )
    else:
        output[:, 0:c_0, :, :] = in_0
    
    # Kernel 2: Interpolate in_1 -> output[:, c_0:c_0+c_1, :, :]
    if h_1 != 40 or w_1 != 40:
        grid_x = (batch_size_1 + 7) // 8
        grid_y = (c_1 + 7) // 8
        grid_z = (40 + BLOCK_SIZE_Y - 1) // BLOCK_SIZE_Y
        
        nearest_interpolate_kernel[(grid_x, grid_y, grid_z)](
            in_1,
            output[:, c_0:c_0+c_1, :, :],
            batch_size_1,
            c_1,
            h_1,
            w_1,
            40,
            40,
            c_1,
            BLOCK_SIZE_X,
            BLOCK_SIZE_Y,
        )
    else:
        output[:, c_0:c_0+c_1, :, :] = in_1
    
    # Kernel 3: Concatenate in_2 and in_3 using Triton kernel
    h_2_3, w_2_3 = h_2, w_2  # Both should be (40, 40) from input specs
    c_total_2_3 = c_2 + c_3
    
    grid_x = (batch_size_2 + 7) // 8
    grid_y = (c_total_2_3 + 7) // 8  
    grid_z = (40 + 7) // 8
    
    concat_kernel[(grid_x, grid_y, grid_z)](
        in_2,
        in_3,
        output[:, c_0+c_1:, :, :],
        batch_size_2,
        c_2,
        c_3,
        40,
        40,
        BLOCK_SIZE_X,
    )
    
    return output

def replacement_func():
    return optimized_interpolate_upstack