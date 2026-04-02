import torch
import triton
import triton.language as tl

def pattern(in_9, in_4, other_args):
    """Match depthwise conv2d + view pattern exactly as in model"""
    # This matches: conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    #              tmp_5 = conv2d.view(1, 512, 64, 64)
    conv2d = torch.conv2d(input=in_9, weight=in_4, groups=512)
    viewed_result = conv2d.view(1, 512, 64, 64)
    return viewed_result

def replacement_args(in_9, in_4, other_args):
    """Extract arguments for the optimized kernel"""
    # We need: input tensor, weight tensor, and input/output shapes info
    input_shape = in_9.shape
    weight_shape = in_4.shape
    return (in_9, in_4, input_shape, weight_shape)

@triton.jit
def depthwise_conv2d_view_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    n_groups,
    input_batch,
    input_channels,
    input_height,
    input_width,
    output_height,
    output_width,
    kernel_height,
    kernel_width,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized depthwise convolution kernel with direct output to correct shape"""
    pid = tl.program_id(0)
    
    # Each thread block handles one group
    if pid >= n_groups:
        return
    
    # Calculate output position
    out_c = pid
    
    # Pointer for this group's input channel
    input_base = input_ptr + pid * input_channels // n_groups * input_height * input_width
    
    # Calculate block of output to compute
    batch_id = pid // (n_groups * output_height * output_width)
    h_out = (pid // (n_groups * output_width)) % output_height
    w_out = pid % output_width
    
    # Calculate input coordinates
    h_in = h_out * stride_h - padding_h
    w_in = w_out * stride_w - padding_w
    
    if h_in < 0 or h_in >= input_height or w_in < 0 or w_in >= input_width:
        # Handle padding - output zero
        tl.store(output_ptr + batch_id * n_groups * output_height * output_width + 
                out_c * output_height * output_width + h_out * output_width + w_out, 0.0)
        return
    
    # Load input patch
    input_offset = input_base + h_in * input_width + w_in
    weight_offset = pid * kernel_height * kernel_width
    
    # Load input and weight values
    input_val = tl.load(input_ptr + input_offset, mask=None)
    weight_val = tl.load(weight_ptr + weight_offset, mask=None)
    
    # Apply convolution (depthwise, so just multiply and sum for single channel)
    result = input_val * weight_val
    
    # Store result directly in target shape [1, n_groups, 64, 64]
    output_offset = batch_id * n_groups * output_height * output_width + \
                   out_c * output_height * output_width + h_out * output_width + w_out
    tl.store(output_ptr + output_offset, result, mask=None)

@torch.fx.wrap
def optimized_depthwise_conv_with_view(in_9, in_4, input_shape, weight_shape):
    """Wrapper for optimized depthwise conv with direct view output"""
    batch, in_channels, in_height, in_width = input_shape
    out_channels, _, kernel_h, kernel_w = weight_shape
    
    # Assume standard convolution parameters for this case
    stride_h, stride_w = 1, 1
    padding_h, padding_w = 3, 3  # Calculated to get from 70x70 to 64x64
    
    output_shape = (1, groups, 64, 64)
    output = torch.zeros(output_shape, dtype=conv_input.dtype, device=conv_input.device)
    
    # Calculate grid size
    total_elements = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3]
    block_size = 256
    grid_size = (total_elements + block_size - 1) // block_size
    
    # Launch kernel
    depthwise_conv2d_view_kernel[grid_size](
        in_9,
        in_4,
        output,
        512,  # Hardcoded groups from pattern
        batch,
        in_channels,
        in_height,
        in_width,
        output_shape[2],
        output_shape[3],
        kernel_h,
        kernel_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        block_size
    )
    
    return output

def replacement_func():
    """Return the optimized function"""
    return optimized_depthwise_conv_with_view