import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    """
    Matches pattern: torch.nn.functional.interpolate(input, size=(width, height), mode='nearest') * multiplier
    for two independent paths.
    """
    # First path: interpolate in_0 to size=(64, 48) and multiply with in_2
    tmp_0 = torch.nn.functional.interpolate(in_0, size=(64, 48), mode='nearest')
    tmp_1 = in_2 * tmp_0
    
    # Second path: interpolate in_1 to size=(32, 24) and multiply with in_3
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(32, 24), mode='nearest')
    tmp_3 = in_3 * tmp_2
    
    return (tmp_1, tmp_3)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def interpolate_multiply_kernel_1(
    input_ptr,
    multiplier_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Kernel for first path: interpolate (nearest) followed by multiplication
    Input: [batch_size, in_channels, in_height, in_width]
    Output: [batch_size, in_channels, out_height, out_width]
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output coordinates
    h_out = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Calculate corresponding input coordinates for nearest neighbor interpolation
    scale_h = in_height / out_height
    scale_w = in_width / out_width
    
    h_in = (h_out * scale_h).to(tl.int32)
    w_in = (w_out * scale_w).to(tl.int32)
    
    # Create masks
    h_out_mask = h_out < out_height
    w_out_mask = w_out < out_width
    mask = h_out_mask[:, None] & w_out_mask[None, :]
    
    # Calculate input addresses
    batch_offset = pid_b * in_channels * in_height * in_width
    channel_offset = pid_c * in_height * in_width
    input_ptr_base = input_ptr + batch_offset + channel_offset
    
    multiplier_offset = pid_b * in_channels + pid_c
    
    # Load input using nearest neighbor access pattern
    input_vals = tl.load(input_ptr_base + h_in[:, None] * in_width + w_in[None, :], 
                        mask=mask, other=0.0)
    
    # Load multiplier (broadcast across spatial dimensions)
    multiplier_val = tl.load(multiplier_ptr + multiplier_offset)
    multiplier_vals = tl.full_like(input_vals, multiplier_val)
    
    # Multiply
    output_vals = input_vals * multiplier_vals
    
    # Calculate output address and store
    output_ptr_base = output_ptr + batch_offset + (pid_c * out_height + h_out) * out_width + w_out
    tl.store(output_ptr_base, output_vals, mask=mask)

@triton.jit
def interpolate_multiply_kernel_2(
    input_ptr,
    multiplier_ptr,
    output_ptr,
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_height,
    out_width,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_W: tl.constexpr,
):
    """
    Kernel for second path: interpolate (nearest) followed by multiplication
    Same logic as kernel_1 but for different input/output dimensions
    """
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)
    
    # Calculate output coordinates
    h_out = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    w_out = pid_w * BLOCK_SIZE_W + tl.arange(0, BLOCK_SIZE_W)
    
    # Calculate corresponding input coordinates for nearest neighbor interpolation
    scale_h = in_height / out_height
    scale_w = in_width / out_width
    
    h_in = (h_out * scale_h).to(tl.int32)
    w_in = (w_out * scale_w).to(tl.int32)
    
    # Create masks
    h_out_mask = h_out < out_height
    w_out_mask = w_out < out_width
    mask = h_out_mask[:, None] & w_out_mask[None, :]
    
    # Calculate input addresses
    batch_offset = pid_b * in_channels * in_height * in_width
    channel_offset = pid_c * in_height * in_width
    input_ptr_base = input_ptr + batch_offset + channel_offset
    
    multiplier_offset = pid_b * in_channels + pid_c
    
    # Load input using nearest neighbor access pattern
    input_vals = tl.load(input_ptr_base + h_in[:, None] * in_width + w_in[None, :], 
                        mask=mask, other=0.0)
    
    # Load multiplier (broadcast across spatial dimensions)
    multiplier_val = tl.load(multiplier_ptr + multiplier_offset)
    multiplier_vals = tl.full_like(input_vals, multiplier_val)
    
    # Multiply
    output_vals = input_vals * multiplier_vals
    
    # Calculate output address and store
    output_ptr_base = output_ptr + batch_offset + (pid_c * out_height + h_out) * out_width + w_out
    tl.store(output_ptr_base, output_vals, mask=mask)

@torch.fx.wrap
def fused_interpolate_multiply(in_0, in_1, in_2, in_3):
    # Get tensor shapes
    shape_0 = in_0.shape
    shape_1 = in_1.shape
    shape_2 = in_2.shape
    shape_3 = in_3.shape
    
    batch_size = shape_0[0]
    in_channels_0 = shape_0[1]
    in_height_0 = shape_0[2]
    in_width_0 = shape_0[3]
    
    in_channels_1 = shape_1[1]
    in_height_1 = shape_1[2]
    in_width_1 = shape_1[3]
    
    # Output sizes from pattern (64, 48) for first path, (32, 24) for second
    out_height_0, out_width_0 = 64, 48
    out_height_1, out_width_1 = 32, 24
    
    # Create output tensors
    output_0 = torch.empty((batch_size, in_channels_0, out_height_0, out_width_0), 
                          dtype=in_0.dtype, device=in_0.device)
    output_1 = torch.empty((batch_size, in_channels_1, out_height_1, out_width_1), 
                          dtype=in_1.dtype, device=in_1.device)
    
    # Configure block sizes
    BLOCK_SIZE_H = 8
    BLOCK_SIZE_W = 8
    
    # Calculate grid configurations
    grid_0 = (
        batch_size,
        in_channels_0,
        (out_height_0 + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (out_width_0 + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    
    grid_1 = (
        batch_size,
        in_channels_1,
        (out_height_1 + BLOCK_SIZE_H - 1) // BLOCK_SIZE_H,
        (out_width_1 + BLOCK_SIZE_W - 1) // BLOCK_SIZE_W,
    )
    
    # Launch kernels for both paths
    interpolate_multiply_kernel_1[grid_0](
        in_0,
        in_2,
        output_0,
        batch_size,
        in_channels_0,
        in_height_0,
        in_width_0,
        out_height_0,
        out_width_0,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    interpolate_multiply_kernel_2[grid_1](
        in_1,
        in_3,
        output_1,
        batch_size,
        in_channels_1,
        in_height_1,
        in_width_1,
        out_height_1,
        out_width_1,
        BLOCK_SIZE_H=BLOCK_SIZE_H,
        BLOCK_SIZE_W=BLOCK_SIZE_W,
    )
    
    return output_0, output_1

def replacement_func():
    return fused_interpolate_multiply