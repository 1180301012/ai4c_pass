import torch
import triton
import triton.language as tl

def pattern(conv_weight, context_input, value_input):
    # Match the conv2d operation with groups=12
    conv2d = torch.conv2d(value_input, conv_weight, None, (1, 1), (32, 0), (1, 1), 12)
    # Match the in-place addition
    context_input += conv2d
    # Return the result that's observable outside
    return context_input

def replacement_args(conv_weight, context_input, value_input):
    return (conv_weight, context_input, value_input)

@triton.jit
def fused_conv2d_add_kernel_group12(
    weight_ptr,          # [12, 1, kernel_h, kernel_w]
    input_ptr,           # [batch, channels_in, height, width]
    output_ptr,          # [batch, channels_out, height, width]
    batch_size,
    in_channels,
    in_height,
    in_width,
    out_channels,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Each program handles one output position
    b = tl.program_id(0)
    c_out = tl.program_id(1)
    h = tl.program_id(2)
    w = tl.program_id(3)
    
    # Check bounds
    if b >= batch_size or c_out >= out_channels or h >= out_height or w >= out_width:
        return
    
    # Calculate input coordinates with padding
    in_h = h * stride_h - pad_h
    in_w = w * stride_w - pad_w
    
    # Calculate output offset
    output_offset = b * out_channels * out_height * out_width + c_out * out_height * out_width + h * out_width + w
    
    # Zero accumulator
    acc = 0.0
    
    # Handle grouped convolution and accumulation
    groups_per_output = out_channels // groups
    group_id = c_out // groups_per_output
    local_c_out = c_out % groups_per_output
    
    # Calculate input channel range for this group
    in_channels_per_group = in_channels // groups
    local_c_in_base = group_id * in_channels_per_group
    
    # Convolution loop
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            for ci in range(in_channels_per_group):
                # Calculate input coordinate with bounds checking
                in_y = in_h + kh
                in_x = in_w + kw
                
                # Skip if out of bounds (padding with zero)
                if in_y < 0 or in_y >= in_height or in_x < 0 or in_x >= in_width:
                    continue
                
                # Calculate offsets
                weight_offset = group_id * kernel_h * kernel_w * in_channels_per_group + kh * kernel_w * in_channels_per_group + kw * in_channels_per_group + ci
                input_offset = b * in_channels * in_height * in_width + (local_c_in_base + ci) * in_height * in_width + in_y * in_width + in_x
                
                # Load weight and input
                weight = tl.load(weight_ptr + weight_offset)
                input_val = tl.load(input_ptr + input_offset)
                
                # Accumulate
                acc += weight * input_val
    
    # Load current output value and add
    current_output = tl.load(output_ptr + output_offset)
    tl.store(output_ptr + output_offset, current_output + acc)

@torch.fx.wrap
def fused_conv2d_add_group12(conv_weight, context_input, value_input):
    weight = conv_weight
    input_tensor = value_input
    
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_h, kernel_w = weight.shape
    
    # Conv2D parameters
    stride_h, stride_w = 1, 1
    pad_h, pad_w = 32, 0
    groups = 12
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - kernel_h) // stride_h + 1
    out_width = (in_width + 2 * pad_w - kernel_w) // stride_w + 1
    
    # Create output tensor
    output = torch.empty_like(context_input)
    
    # Set up grid dimensions
    grid = (
        batch_size,
        out_channels,
        out_height,
        out_width,
    )
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Launch kernel
    fused_conv2d_add_kernel_group12[grid](
        weight_ptr=weight,
        input_ptr=input_tensor,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        out_channels=out_channels,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        pad_h=pad_h,
        pad_w=pad_w,
        groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_conv2d_add_group12