import torch
import triton
import triton.language as tl

def pattern(conv_weight, residual_input, conv_input):
    # Match: Conv2D -> Addition -> Interpolation sequence
    conv_out = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)
    add_out = residual_input + conv_out
    interpolate_out = torch.nn.functional.interpolate(add_out, (64, 64), None, 'bilinear', False)
    return interpolate_out

def replacement_args(conv_weight, residual_input, conv_input):
    return (conv_weight, residual_input, conv_input)

@triton.jit
def conv_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program IDs
    batch_id = tl.program_id(0)
    out_channel_id = tl.program_id(1)
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Calculate input address offset
    output_offset = batch_id * out_channels * output_height * output_width + \
                   out_channel_id * output_height * output_width + \
                   spatial_offset
    
    # Bounds checking
    spatial_mask = spatial_offset < output_width
    
    # Load input data for this output position
    # Since this is 1x1 convolution, each output pixel depends on all input channels
    input_base = (batch_id * in_channels * input_height * input_width + 
                  spatial_offset)
    
    # For 1x1 convolution, we need to sum over all input channels
    summed_val = 0.0
    for in_c in range(in_channels):
        input_ptr_offset = input_base + in_c * input_height * input_width
        input_val = tl.load(input_ptr + input_ptr_offset, mask=spatial_mask, other=0.0).to(tl.float32)
        weight_ptr_offset = out_channel_id * in_channels + in_c
        weight_val = tl.load(weight_ptr + weight_ptr_offset, mask=spatial_mask & tl.arange(0, 1), other=0.0).to(tl.float32)
        summed_val += input_val * weight_val
    
    tl.store(output_ptr + output_offset, summed_val, mask=spatial_mask)

@triton.jit
def add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@triton.jit
def interpolate_kernel(
    input_ptr,
    output_ptr,
    batch_size,
    channels,
    input_height,
    input_width,
    output_height,
    output_width,
    BLOCK_SIZE: tl.constexpr,
):
    batch_id = tl.program_id(0)
    channel_id = tl.program_id(1)
    spatial_offset = tl.program_id(2) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Convert spatial offset to 2D coordinates
    output_y = spatial_offset // output_width
    output_x = spatial_offset % output_width
    
    # Bounds checking
    spatial_mask = spatial_offset < output_height * output_width
    
    # Calculate input coordinates (normalized)
    x_scale = (input_width - 1) / max(output_width - 1, 1)
    y_scale = (input_height - 1) / max(output_height - 1, 1)
    
    input_x = output_x * x_scale
    input_y = output_y * y_scale
    
    # Calculate weights and positions for bilinear interpolation
    x0 = tl.floor(input_x).to(tl.int32)
    y0 = tl.floor(input_y).to(tl.int32)
    
    # Ensure coordinates are within bounds
    x0 = tl.max(x0, 0).to(tl.int32)
    y0 = tl.max(y0, 0).to(tl.int32)
    x0 = tl.min(x0, input_width - 1).to(tl.int32)
    y0 = tl.min(y0, input_height - 1).to(tl.int32)
    
    # Calculate interpolation weights
    fx = input_x - x0
    fy = input_y - y0
    
    # Get neighboring coordinates
    x1 = tl.min(x0 + 1, input_width - 1).to(tl.int32)
    y1 = tl.min(y0 + 1, input_height - 1).to(tl.int32)
    
    # Base address for this batch and channel
    input_base = batch_id * channels * input_height * input_width + \
                 channel_id * input_height * input_width
    
    # Calculate addresses for the four corner points
    addr_00 = input_base + y0 * input_width + x0
    addr_01 = input_base + y1 * input_width + x0
    addr_10 = input_base + y0 * input_width + x1
    addr_11 = input_base + y1 * input_width + x1
    
    # Load the four corner values
    q00 = tl.load(input_ptr + addr_00, mask=spatial_mask, other=0.0).to(tl.float32)
    q01 = tl.load(input_ptr + addr_01, mask=spatial_mask, other=0.0).to(tl.float32)
    q10 = tl.load(input_ptr + addr_10, mask=spatial_mask, other=0.0).to(tl.float32)
    q11 = tl.load(input_ptr + addr_11, mask=spatial_mask, other=0.0).to(tl.float32)
    
    # Perform bilinear interpolation
    top = (1-fx) * q00 + fx * q10
    bottom = (1-fx) * q01 + fx * q11
    val = (1-fy) * top + fy * bottom
    
    # Calculate output address and store result
    output_addr = (batch_id * channels * output_height * output_width + 
                  channel_id * output_height * output_width + 
                  spatial_offset)
    
    tl.store(output_ptr + output_addr, val, mask=spatial_mask)

@torch.fx.wrap
def optimized_conv_add_interpolate(conv_weight, residual_input, conv_input):
    batch_size, in_channels, input_h, input_w = conv_input.shape
    out_channels = conv_weight.shape[0]
    
    # Step 1: Conv2D (1x1)
    conv_out = torch.empty(batch_size, out_channels, input_h, input_w, 
                           device=conv_input.device, dtype=conv_input.dtype)
    
    grid_z = batch_size
    grid_y = out_channels
    grid_x = (input_w * input_h + 256 - 1) // 256
    
    conv_kernel[(
        grid_z, grid_y, grid_x
    )](
        input_ptr=conv_input,
        weight_ptr=conv_weight,
        output_ptr=conv_out,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_height=input_h,
        input_width=input_w,
        output_height=input_h,
        output_width=input_w,
        BLOCK_SIZE=256
    )
    
    # Step 2: Addition
    add_out = torch.empty_like(conv_out)
    total_elements = batch_size * out_channels * input_h * input_w
    num_programs = (total_elements + 256 - 1) // 256
    add_kernel[(num_programs,)](
        conv_out, residual_input, add_out, total_elements, 256
    )
    
    # Step 3: Interpolate
    interpolate_out = torch.empty(batch_size, out_channels, 64, 64,
                                 device=add_out.device, dtype=add_out.dtype)
    
    grid_z = batch_size
    grid_y = out_channels
    grid_x = (64 * 64 + 256 - 1) // 256
    
    interpolate_kernel[(
        grid_z, grid_y, grid_x
    )](
        add_out, interpolate_out,
        batch_size, out_channels,
        input_h, input_w, 64, 64,
        BLOCK_SIZE=256
    )
    
    return interpolate_out

def replacement_func():
    return optimized_conv_add_interpolate