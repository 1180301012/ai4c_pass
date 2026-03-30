import torch
import triton
import triton.language as tl

def pattern(input_tensor, conv_weights, bn_running_mean, bn_running_var, bn_weight, bn_bias, add_tensor):
    # Conv2D operation
    conv2d = torch.conv2d(input_tensor, conv_weights, None, (1, 1), (1, 1), (1, 1), 1)
    
    # Batch normalization operation
    tmp_6 = torch.nn.functional.batch_norm(conv2d, bn_running_mean, bn_running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    
    # Leaky ReLU operation
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    
    # Return the Leaky ReLU output (addition will be handled separately)
    return tmp_7

def replacement_args(input_tensor, conv_weights, bn_running_mean, bn_running_var, bn_weight, bn_bias, add_tensor):
    return (input_tensor, conv_weights, bn_running_mean, bn_running_var, bn_weight, bn_bias, add_tensor)

@triton.jit
def fused_conv_bn_relu_kernel(
    input_ptr,
    conv_weight_ptr,
    bn_running_mean_ptr,
    bn_running_var_ptr,
    bn_weight_ptr, 
    bn_bias_ptr,
    output_ptr,
    n_channels,
    height,
    width,
    batch_size,
    in_channels,
    kh,
    kw,
    stride_h: tl.constexpr,
    stride_w: tl.constexpr,
    pad_h: tl.constexpr,
    pad_w: tl.constexpr,
    dilation_h: tl.constexpr,
    dilation_w: tl.constexpr,
    groups: tl.constexpr,
    weight_running_mean_ptr,
    weight_running_var_ptr,
    eps: tl.constexpr,
    leaky_relu_neg_slope: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # Extract meta pointers from weight batch for this specific channel
    # Each program processes a slice of the output channels
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)  # batch dimension
    
    # Determine what this program computes
    m_offset = pid_m * BLOCK_SIZE_M
    n_offset = pid_n * BLOCK_SIZE_N
    b_offset = pid_b * batch_size  # Assuming batch_size processing per program
    
    # Load batch normalization parameters for this output channel range
    if bn_running_mean_ptr is not None:
        bn_mean = tl.load(bn_running_mean_ptr + n_offset, mask=(n_offset < n_channels), other=0.0)
        bn_var = tl.load(bn_running_var_ptr + n_offset, mask=(n_offset < n_channels), other=1.0)
        bn_weight_val = tl.load(bn_weight_ptr + n_offset, mask=(n_offset < n_channels), other=1.0)
        bn_bias_val = tl.load(bn_bias_ptr + n_offset, mask=(n_offset < n_channels), other=0.0)
        
        # Compute batch normalization scale and shift
        bn_scale = bn_weight_val / tl.sqrt(bn_var + eps)
        bn_shift = bn_bias_val - bn_mean * bn_scale
    else:
        bn_scale = 1.0
        bn_shift = 0.0
    
    # Compute convolution output for this position range
    # Simplified convolution pattern for 1x1 conv with stride 1 and padding 1
    h_start = 0
    h_end = height
    w_start = 0
    w_end = width
    
    # Load input data - process spatial positions efficiently
    spatial_positions = m_offset * height * width + n_offset * 1  # Channel-major layout
    base_offset = b_offset * height * width * in_channels + spatial_positions
    
    if input_ptr is not None and h_start < h_end and w_start < w_end:
        # Load input for this batch and spatial region
        input_base = tl.load(input_ptr + base_offset, mask=(pid_b < batch_size), other=0.0)
        input_val = input_base
    else:
        input_val = 0.0
    
    # Load convolution weights (simplified for 1x1 conv)
    weight_offset = n_offset * in_channels * kh * kw + m_offset * 1
    if conv_weight_ptr is not None:
        weight_data = tl.load(conv_weight_ptr + weight_offset, mask=(n_offset < n_channels and m_offset < batch_size), other=0.0)
        weight_val = weight_data
    else:
        weight_val = 0.0
    
    # Perform convolution, batch normalization, and activation in one step
    conv_output = input_val * weight_val
    bn_output = conv_output * bn_scale + bn_shift
    relu_output = tl.where(bn_output > 0, bn_output, bn_output * leaky_relu_neg_slope)
    
    # Store output
    output_base = pid_b * n_channels * height * width + m_offset * height * width + h_start * width + w_start
    if output_ptr is not None:
        tl.store(output_ptr + output_base, relu_output, mask=(pid_b < batch_size and n_offset < n_channels))

@torch.fx.wrap
def fused_conv_bn_relu(input_tensor, conv_weights, bn_running_mean, bn_running_var, bn_weight, bn_bias):
    # Get tensor shapes
    batch_size, in_channels, height, width = input_tensor.shape
    out_channels = conv_weights.shape[0]
    
    # Create output tensor
    output = torch.empty((batch_size, out_channels, height, width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Set up grid dimensions
    BLOCK_SIZE_M = 32  # Output channels per program
    BLOCK_SIZE_N = 128  # Input channels per program
    BLOCK_SIZE_K = 32   # Spatial dimensions
    
    # Grid dimensions: (output_channels // BLOCK_SIZE_M, input_channels // BLOCK_SIZE_N, batch_size)
    grid_x = (out_channels + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
    grid_y = (in_channels + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N
    grid_z = batch_size
    
    # Determine meta pointers
    bn_mean_ptr = bn_running_mean.data_ptr() if bn_running_mean is not None else None
    bn_var_ptr = bn_running_var.data_ptr() if bn_running_var is not None else None
    bn_weight_ptr = bn_weight.data_ptr() if bn_weight is not None else None
    bn_bias_ptr = bn_bias.data_ptr() if bn_bias is not None else None
    
    # Launch kernel
    fused_conv_bn_relu_kernel[(grid_x, grid_y, grid_z)](
        input_tensor.data_ptr(),
        conv_weights.data_ptr(),
        bn_mean_ptr,
        bn_var_ptr,
        bn_weight_ptr,
        bn_bias_ptr,
        output.data_ptr(),
        out_channels,
        height,
        width,
        batch_size,
        in_channels,
        conv_weights.shape[2],  # kernel height
        conv_weights.shape[3],  # kernel width
        1, 1,  # stride_h, stride_w
        1, 1,  # pad_h, pad_w
        1, 1,  # dilation_h, dilation_w
        1,  # groups
        None, None,  # weight_running_mean, weight_running_var
        1e-05,  # epsilon
        0.01,  # leaky_relu_negative_slope
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
        8  # GROUP_SIZE_M
    )
    
    return output

def replacement_func():
    return fused_conv_bn_relu