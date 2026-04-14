import torch
import triton
import triton.language as tl

@triton.jit
def fused_conv2d_relu_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    batch_size,
    in_channels,
    out_channels,
    height,
    width,
    weight_height,
    weight_width,
    stride_height,
    stride_width,
    pad_height,
    pad_width,
    dilation_height,
    dilation_width,
    n_groups,
    BLOCK_SIZE: tl.constexpr,
):
    # Grid configuration based on output dimensions
    pid = tl.program_id(0)
    
    # Calculate output coordinate
    pid_x = pid % width
    pid_y = (pid // width) % height
    pid_b = (pid // (width * height)) % batch_size
    pid_c = pid // (width * height * batch_size)
    
    if pid_c >= out_channels:
        return
    
    # Calculate input coordinate with padding
    in_y = pid_y * stride_height - pad_height
    in_x = pid_x * stride_width - pad_width
    
    # Initialize accumulator
    accumulator = 0.0
    if bias_ptr is not None:
        accumulator = tl.load(bias_ptr + pid_c)
    
    # Convolution computation
    for c_in in range(0, in_channels, n_groups):
        c_in_max = min(c_in + n_groups, in_channels)
        
        for ky in range(0, weight_height):
            for kx in range(0, weight_width):
                # Calculate input coordinate with dilation
                input_y = in_y + ky * dilation_height
                input_x = in_x + kx * dilation_width
                
                # Check bounds
                if (0 <= input_y < height and 0 <= input_x < width and
                    c_in < in_channels):
                    
                    # Calculate memory addresses
                    input_idx = (pid_b * in_channels + c_in) * (height * width) + input_y * width + input_x
                    weight_idx = (pid_c * in_channels + c_in) * (weight_height * weight_width) + ky * weight_width + kx
                    
                    input_val = tl.load(input_ptr + input_idx, other=0.0)
                    weight_val = tl.load(weight_ptr + weight_idx)
                    
                    accumulator += input_val * weight_val
    
    # Apply ReLU
    output_val = tl.maximum(accumulator, 0.0)
    
    # Store result
    output_idx = (pid_b * out_channels + pid_c) * (height * width) + pid_y * width + pid_x
    tl.store(output_ptr + output_idx, output_val)

@torch.fx.wrap
def fused_conv2d_relu(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    # Get input dimensions
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, weight_height, weight_width = weight_tensor.shape
    stride_height, stride_width = stride
    pad_height, pad_width = padding
    dilation_height, dilation_width = dilation
    
    # Calculate output dimensions
    out_height = (in_height + 2 * pad_height - dilation_height * (weight_height - 1) - 1) // stride_height + 1
    out_width = (in_width + 2 * pad_width - dilation_width * (weight_width - 1) - 1) // stride_width + 1
    
    # Create output tensor
    output_shape = (batch_size, out_channels, out_height, out_width)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate total elements and launch configuration
    total_elements = batch_size * out_channels * out_height * out_width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    fused_conv2d_relu_kernel[(num_programs,)](
        input_ptr=input_tensor,
        weight_ptr=weight_tensor,
        bias_ptr=bias_tensor if bias_tensor is not None else None,
        output_ptr=output,
        batch_size=batch_size,
        in_channels=in_channels,
        out_channels=out_channels,
        height=out_height,
        width=out_width,
        weight_height=weight_height,
        weight_width=weight_width,
        stride_height=stride_height,
        stride_width=stride_width,
        pad_height=pad_height,
        pad_width=pad_width,
        dilation_height=dilation_height,
        dilation_width=dilation_width,
        n_groups=groups,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def pattern(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    conv2d_result = torch.conv2d(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)
    relu_result = torch.nn.functional.relu(conv2d_result, inplace=True)
    return relu_result

def replacement_args(input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups):
    return (input_tensor, weight_tensor, bias_tensor, stride, padding, dilation, groups)

def replacement_func():
    return fused_conv2d_relu