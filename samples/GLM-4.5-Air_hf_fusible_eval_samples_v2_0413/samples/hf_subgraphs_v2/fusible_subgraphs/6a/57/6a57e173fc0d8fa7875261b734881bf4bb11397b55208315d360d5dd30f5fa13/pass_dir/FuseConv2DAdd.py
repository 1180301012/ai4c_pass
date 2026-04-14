import torch
import triton
import triton.language as tl

# Pattern matching for conv2d + add operations
def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_0, None, (1, 1), (32, 0), (1, 1), 4)
    result = in_1 + conv2d
    return result

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

# Optimized Triton kernel for fused conv2d + add
@triton.jit
def fused_conv2d_add_kernel(
    input_ptr,          # in_2 (value layer)
    weight_ptr,         # in_0 (conv weights)  
    bias_ptr,           # None
    output_ptr,         # result buffer
    batch_size,
    in_channels,
    in_height,
    in_width,
    weight_channels,
    weight_height,
    weight_width,
    out_channels,
    stride_h, stride_w,
    pad_h, pad_w,
    dilation_h, dilation_w,
    groups,
    
    # Triton-specific parameters
    BLOCK_SIZE: tl.constexpr,
):
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_h - dilation_h * (weight_height - 1) - 1) // stride_h + 1
    out_width = (in_width + 2 * pad_w - dilation_w * (weight_width - 1) - 1) // stride_w + 1
    
    # Get program ID
    pid = tl.program_id(axis=0)
    
    # Calculate output position
    out_h = pid // out_width
    out_w = pid % out_width
    
    # Ensure we're within bounds
    if out_h >= out_height or out_w >= out_width:
        return
        
    # Process each group and output channel within the group
    for c in range(out_channels // groups):
        for g in range(groups):
            weight_offset = g * weight_channels + c
            
            # Calculate input positions for this group
            in_c_base = g * (in_channels // groups)
            
            # Load bias (if exists, but it's None in our case)
            if bias_ptr is not None:
                bias_val = tl.load(bias_ptr + c + g * out_channels)
            else:
                bias_val = 0.0
                
            acc = bias_val
            
            # Perform convolution for this position
            for kh in range(weight_height):
                for kw in range(weight_width):
                    # Calculate input coordinates with padding and dilation
                    in_h = out_h * stride_h - pad_h + kh * dilation_h
                    in_w = out_w * stride_w - pad_w + kw * dilation_w
                    
                    # Check bounds
                    if 0 <= in_h < in_height and 0 <= in_w < in_width:
                        # Load weight value
                        weight_val = tl.load(weight_ptr + weight_offset * (weight_height * weight_width) + kh * weight_width + kw)
                        
                        # Load input value and accumulate
                        for ic in range(in_channels // groups):
                            input_offset = ((batch_size * in_channels + in_c_base + ic) * in_height + in_h) * in_width + in_w
                            input_val = tl.load(input_ptr + input_offset)
                            acc += weight_val * input_val
            
            # Store result
            out_offset = ((batch_size * out_channels + g * (out_channels // groups) + c) * out_height + out_h) * out_width + out_w
            tl.store(output_ptr + out_offset, acc)

@torch.fx.wrap
def fused_conv2d_add(in_0, in_1, in_2):
    # Get tensor shapes
    in_2_shape = in_2.shape  # [batch, channels, height, width]
    in_0_shape = in_0.shape  # [out_channels, channels/groups, kernel_h, kernel_w]
    
    batch_size = in_2_shape[0]
    in_channels = in_2_shape[1]
    in_height = in_2_shape[2]
    in_width = in_2_shape[3]
    
    out_channels = in_0_shape[0]
    weight_height = in_0_shape[2]
    weight_width = in_0_shape[3]
    
    # Output shape (calculated from conv2d parameters)
    out_height = (in_height + 2*32 - 1*(weight_height-1) - 1) // 1 + 1  # pad_h=32, stride_h=1, dilation_h=1
    out_width = (in_width + 2*0 - 1*(weight_width-1) - 1) // 1 + 1     # pad_w=0, stride_w=1, dilation_w=1
    
    # Create output tensor
    output_shape = [batch_size, out_channels, out_height, out_width]
    output = torch.empty(output_shape, dtype=in_2.dtype, device=in_2.device)
    
    # Launch kernel
    total_output_elements = batch_size * out_channels * out_height * out_width
    BLOCK_SIZE = 1024
    num_programs = (total_output_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_add_kernel[(num_programs,)](
        input_ptr=in_2,
        weight_ptr=in_0,
        bias_ptr=None,  # No bias in our case
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        in_height=in_height,
        in_width=in_width,
        weight_channels=out_channels // 4,  # groups=4
        weight_height=weight_height,
        weight_width=weight_width,
        out_channels=out_channels,
        stride_h=1, stride_w=1,
        pad_h=32, pad_w=0,
        dilation_h=1, dilation_w=1,
        groups=4,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Add to in_1 (which has same shape as output)
    result = in_1 + output
    return result

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv2d_add