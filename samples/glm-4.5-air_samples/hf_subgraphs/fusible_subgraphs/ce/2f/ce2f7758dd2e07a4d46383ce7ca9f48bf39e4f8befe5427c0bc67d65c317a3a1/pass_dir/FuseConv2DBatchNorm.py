import torch
import triton
import triton.language as tl

def pattern(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, pool_input):
    # Conv2D operation (using in_6 as input, in_4 as weight)
    tmp_5 = torch.conv2d(conv_input, conv_weight, None, (1, 1), (1, 1), (1, 1), 1)
    
    # BatchNorm operation (using tmp_5 as input)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Return intermediate and final results - tmp_6 is the fused conv+bn result
    return tmp_6

def replacement_args(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, pool_input):
    return (conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, pool_input)

@triton.jit
def fused_conv_bn_kernel(
    x_ptr,  # input [batch_size, in_channels, input_h, input_w]
    w_ptr,  # weight [out_channels, in_channels, 1, 1]
    running_mean_ptr,  # [out_channels]
    running_var_ptr,  # [out_channels]
    weight_bn_ptr,  # [out_channels]
    bias_bn_ptr,  # [out_channels]
    y_ptr,  # output [batch_size, out_channels, input_h, input_w]
    batch_size: tl.constexpr,
    in_channels: tl.constexpr,
    out_channels: tl.constexpr,
    input_h: tl.constexpr,
    input_w: tl.constexpr
):
    pid = tl.program_id(0)
    channels_per_program = tl.cdiv(out_channels, 1)  # Each program handles full channel range
    out_channel_start = pid * channels_per_program
    out_channel_end = min(out_channel_start + channels_per_program, out_channels)
    
    # Only proceed if we have valid output channels
    if out_channel_start >= out_channels:
        return
        
    # Load batch normalization parameters
    bn_weight = tl.load(weight_bn_ptr + out_channel_start, out_channel_start < out_channels, other=1.0).to(tl.float32)
    bn_bias = tl.load(bias_bn_ptr + out_channel_start, out_channel_start < out_channels, other=0.0).to(tl.float32)
    running_mean = tl.load(running_mean_ptr + out_channel_start, out_channel_start < out_channels, other=0.0).to(tl.float32)
    running_var = tl.load(running_var_ptr + out_channel_start, out_channel_start < out_channels, other=1.0).to(tl.float32)
    
    # Compute inverse standard deviation
    inv_std = 1.0 / tl.sqrt(running_var + 1e-05)
    
    # Process each spatial location independently (simple 1x1 conv)
    for h in range(input_h):
        for w in range(input_w):
            # Process each batch
            for b in range(batch_size):
                # Compute convolution output for this batch and spatial location
                conv_output = 0.0
                for c_in in range(in_channels):
                    # Load input value: x[b, c_in, h, w]
                    x_offset = b * in_channels * input_h * input_w + c_in * input_h * input_w + h * input_w + w
                    x_val = tl.load(x_ptr + x_offset, other=0.0).to(tl.float32)
                    
                    # Load weight: w[out_channel_start, c_in, 0, 0] (1x1 conv)
                    w_offset = out_channel_start * in_channels + c_in
                    w_val = tl.load(w_ptr + w_offset, other=0.0).to(tl.float32)
                    
                    conv_output += x_val * w_val
                
                # Apply batch normalization: (x - running_mean) * inv_std * weight + bias
                normalized_output = conv_output * inv_std + bn_bias + running_mean
                
                # Store output: y[b, out_channel_start, h, w]
                y_offset = b * out_channels * input_h * input_w + out_channel_start * input_h * input_w + h * input_w + w
                tl.store(y_ptr + y_offset, normalized_output)

@torch.fx.wrap
def fused_conv_bn(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    # Get tensor shapes
    batch_size, in_channels, input_h, input_w = conv_input.shape
    # conv_weight shape: [out_channels, in_channels, 1, 1] -> extract out_channels
    out_channels = conv_weight.shape[0]
    
    # Launch with one program per output channel (simple approach)
    num_programs = out_channels
    
    # Create output tensor
    output_shape = (batch_size, out_channels, input_h, input_w)
    output = torch.empty(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Launch kernel
    fused_conv_bn_kernel[(num_programs,)](
        x_ptr=conv_input,
        w_ptr=conv_weight,
        running_mean_ptr=bn_running_mean,
        running_var_ptr=bn_running_var,
        weight_bn_ptr=bn_weight,
        bias_bn_ptr=bn_bias,
        y_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        input_h=input_h,
        input_w=input_w
    )
    
    return output

@torch.fx.wrap  
def optimized_forward(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias, pool_input):
    # Fuse conv2d + batch_norm only
    fused_result = fused_conv_bn(conv_input, conv_weight, bn_running_mean, bn_running_var, bn_weight, bn_bias)
    
    return fused_result

def replacement_func():
    return optimized_forward