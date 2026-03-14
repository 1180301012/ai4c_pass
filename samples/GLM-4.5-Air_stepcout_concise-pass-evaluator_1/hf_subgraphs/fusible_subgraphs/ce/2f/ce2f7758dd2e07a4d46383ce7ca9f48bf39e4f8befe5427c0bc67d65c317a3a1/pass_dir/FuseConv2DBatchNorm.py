import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Variable assignments exactly as in the original model
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = in_4
    
    # Conv2D operation with the exact signature from the model
    tmp_5 = torch.conv2d(in_6, tmp_4, None, (1, 1), (1, 1), (1, 1), 1)
    
    # BatchNorm operation with the exact signature from the model  
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    
    # Need to return inputs that are used outside the subgraph for compatibility
    return in_5, tmp_6

def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)

@triton.jit
def fused_conv_bn_kernel(
    input_ptr, weight_ptr, bn_weight_ptr, bn_bias_ptr, running_mean_ptr, running_var_ptr,
    output_ptr, n_elements, 
    batch_size, in_channels, out_channels, 
    input_h, input_w, weight_h, weight_w,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate output spatial dimensions (for 1x1 conv with padding=1, stride=1, dilation=1)
    output_h = input_h
    output_w = input_w
    
    # For each position in output tensor
    output_offset = offsets
    
    # Decompose offset into batch, channel, h, w coordinates
    output_channel = output_offset // (batch_size * output_h * output_w)
    remaining = output_offset % (batch_size * output_h * output_w)
    batch_idx = remaining // (output_h * output_w)
    spatial_offset = remaining % (output_h * output_w)
    h_idx = spatial_offset // output_w
    w_idx = spatial_offset % output_w
    
    # Load input data (with border handling for 1x1 conv + padding=1)
    input_val = tl.zeros((in_channels,), dtype=tl.float32)
    for c in range(in_channels):
        input_h_idx = h_idx - 1  # padding = 1
        input_w_idx = w_idx - 1  # padding = 1
        
        if 0 <= input_h_idx < input_h and 0 <= input_w_idx < input_w:
            input_offset = batch_idx * (in_channels * input_h * input_w) + c * (input_h * input_w) + input_h_idx * input_w + input_w_idx
            input_val[c] = tl.load(input_ptr + input_offset, mask=True)
    
    # Load weight for current output channel
    weight_val = tl.zeros((in_channels,), dtype=tl.float32)
    for c in range(in_channels):
        weight_offset = output_channel * (in_channels * weight_h * weight_w) + c * (weight_h * weight_w) + 1 * weight_w + 1  # center of 3x3 kernel
        weight_val[c] = tl.load(weight_ptr + weight_offset, mask=True)
    
    # Convolution operation
    conv_val = 0.0
    for c in range(in_channels):
        conv_val += input_val[c] * weight_val[c]
    
    # Load batch norm parameters
    running_mean_val = tl.load(running_mean_ptr + output_channel, mask=True)
    running_var_val = tl.load(running_var_ptr + output_channel, mask=True)
    bn_weight_val = tl.load(bn_weight_ptr + output_channel, mask=True)
    bn_bias_val = tl.load(bn_bias_ptr + output_channel, mask=True)
    
    # Batch normalization
    sqrt_var = tl.sqrt(running_var_val + 1e-05)
    normalized_val = (conv_val - running_mean_val) / sqrt_var
    bn_out_val = normalized_val * bn_weight_val + bn_bias_val
    
    # Store output
    tl.store(output_ptr + output_offset, bn_out_val, mask=mask)

@torch.fx.wrap
def fused_conv_bn(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    # Get input shapes for conv2d (in_6 is the input tensor, in_4 is the weight)
    batch_size, in_channels, input_h, input_w = in_6.shape
    out_channels, _, weight_h, weight_w = in_4.shape
    
    # Calculate output shape (1x1 conv with stride=1, padding=1, dilation=1)
    output_h = input_h
    output_w = input_w
    
    # Calculate total elements
    n_elements = batch_size * out_channels * output_h * output_w
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, output_h, output_w), dtype=in_6.dtype, device=in_6.device)
    
    # Choose block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid_size = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_conv_bn_kernel[grid_size](
        input_ptr=in_6,
        weight_ptr=in_4,
        bn_weight_ptr=in_3,
        bn_bias_ptr=in_2,
        running_mean_ptr=in_0,
        running_var_ptr=in_1,
        output_ptr=output,
        n_elements=n_elements,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=input_h,
        input_w=input_w,
        weight_h=weight_h,
        weight_w=weight_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return in_5, output  # Return in_5 (unchanged) and fused conv-bn output

def replacement_func():
    return fused_conv_bn