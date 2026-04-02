import torch
import triton
import triton.language as tl
import math

def pattern(conv_input, conv_weight, conv_bias, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    """
    Pattern to match: Conv2D + BatchNorm fusion
    """
    conv_result = torch.conv2d(conv_input, conv_weight, conv_bias, (1, 1), (1, 1), (1, 1), 1)
    bn_result = torch.nn.functional.batch_norm(conv_result, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    return conv_result, bn_result

def replacement_args(conv_input, conv_weight, conv_bias, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    return (conv_input, conv_weight, conv_bias, bn_running_mean, bn_running_var, bn_weight, bn_bias)

@triton.jit
def fused_conv_bn_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    conv_output_ptr,
    bn_output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    kernel_size_h,
    kernel_size_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    eps: tl.constexpr,
):
    """Fused Conv2D + BatchNorm kernel - simplified for single block processing"""
    pid = tl.program_id(0)
    
    # Calculate output tensor dimensions
    out_height = (height + 2 * pad_h - kernel_size_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_size_w) // stride_w + 1
    
    # Each program processes one output location for all channels
    out_y = pid // out_width
    out_x = pid % out_width
    
    if out_y >= out_height or out_x >= out_width:
        return
    
    # Load batch norm parameters
    bn_weight_vec = tl.load(bn_weight_ptr + tl.arange(0, out_channels, dtype=tl.int32), mask=tl.arange(out_channels) < out_channels)
    bn_bias_vec = tl.load(bn_bias_ptr + tl.arange(0, out_channels, dtype=tl.int32), mask=tl.arange(out_channels) < out_channels)
    running_mean_vec = tl.load(running_mean_ptr + tl.arange(0, out_channels, dtype=tl.int32), mask=tl.arange(out_channels) < out_channels)
    running_var_vec = tl.load(running_var_ptr + tl.arange(0, out_channels, dtype=tl.int32), mask=tl.arange(out_channels) < out_channels)
    
    # Precompute normalization factors
    inv_std = tl.sqrt(running_var_vec + eps)
    scale = bn_weight_vec * inv_std
    bias = bn_bias_vec - running_mean_vec * scale
    
    # Compute convolution for current output location
    conv_results = tl.zeros((out_channels,), dtype=tl.float32)
    
    # Input coordinates with padding
    in_y_start = out_y * stride_h - pad_h
    in_x_start = out_x * stride_w - pad_w
    
    for c_out in range(out_channels):
        conv_val = 0.0
        for kh in range(kernel_size_h):
            for kw in range(kernel_size_w):
                in_y_idx = in_y_start + kh
                in_x_idx = in_x_start + kw
                
                if in_y_idx >= 0:
                    if in_y_idx < height:
                        if in_x_idx >= 0:
                            if in_x_idx < width:
                                for c_in in range(in_channels):
                                    # Calculate linear indices
                                    input_idx = c_in * height * width + in_y_idx * width + in_x_idx
                                    weight_idx = (c_out * in_channels + c_in) * kernel_size_h * kernel_size_w + kh * kernel_size_w + kw
                                    
                                    input_val = tl.load(input_ptr + input_idx * 4)
                                    weight_val = tl.load(weight_ptr + weight_idx * 4)
                                    conv_val += input_val * weight_val
        
        # Add bias if provided
        if bias_ptr is not None:
            bias_val = tl.load(bias_ptr + c_out * 4)
            conv_val += bias_val
        
        conv_results[c_out] = conv_val
    
    # Apply batch normalization and store results
    bn_results = conv_results * scale + bias
    
    # Store conv and bn outputs
    out_idx_base = out_y * out_width + out_x
    for c_out in range(out_channels):
        output_idx = out_idx_base + c_out * out_height * out_width
        
        # Store convolution result
        tl.store(conv_output_ptr + output_idx * 4, conv_results[c_out])
        # Store batch norm result  
        tl.store(bn_output_ptr + output_idx * 4, bn_results[c_out])

@torch.fx.wrap
def fused_conv_bn(input_tensor, weight_tensor, bias_tensor, running_mean, running_var, bn_weight, bn_bias, stride, padding):
    """Fused Conv2D + BatchNorm wrapper"""
    # Get input dimensions (simplified for single batch case)
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels, _, kernel_size_h, kernel_size_w = weight_tensor.shape
    
    # Calculate output dimensions
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    out_height = (height + 2 * pad_h - kernel_size_h) // stride_h + 1
    out_width = (width + 2 * pad_w - kernel_size_w) // stride_w + 1
    
    # Prepare output tensors
    conv_output = torch.zeros((batch_size, out_channels, out_height, out_width), dtype=input_tensor.dtype, device=input_tensor.device)
    bn_output = torch.zeros((batch_size, out_channels, out_height, out_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Flatten outputs for kernel processing
    conv_output_flat = conv_output.view(-1)
    bn_output_flat = bn_output.view(-1)
    
    # Launch kernel - one program per output location (H x W for each channel)
    total_output_locations = out_height * out_width
    grid = (total_output_locations,)
    
    # Handle null bias tensor (bias=None in original)
    if bias_tensor is None:
        bias_ptr = None
    else:
        bias_ptr = bias_tensor
    
    fused_conv_bn_kernel[grid](
        input_tensor,
        weight_tensor,
        bias_ptr,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        conv_output_flat,
        bn_output_flat,
        batch_size,
        in_channels,
        out_channels,
        height,
        width,
        kernel_size_h,
        kernel_size_w,
        stride_h,
        stride_w,
        pad_h,
        pad_w,
        1e-05  # epsilon from original
    )
    
    return conv_output, bn_output

def replacement_func():
    return fused_conv_bn