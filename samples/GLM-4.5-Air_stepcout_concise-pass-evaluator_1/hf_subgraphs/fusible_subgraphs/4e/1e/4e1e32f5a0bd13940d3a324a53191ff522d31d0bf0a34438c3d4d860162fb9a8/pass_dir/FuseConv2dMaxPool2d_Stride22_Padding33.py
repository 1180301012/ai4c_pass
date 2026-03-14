import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight_tensor):
    """
    Match Conv2d + MaxPool2d pattern with stride (2,2) and padding (3,3)
    """
    conv_output = torch.conv2d(input_tensor, weight_tensor, None, (2, 2), (3, 3), (1, 1), 1)
    maxpool_output = torch.nn.functional.max_pool2d(conv_output, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    return maxpool_output

def replacement_args(input_tensor, weight_tensor):
    return (input_tensor, weight_tensor, (2, 2), (3, 3), (1, 1), 1, 3)

@triton.jit
def fused_conv2d_maxpool_kernel(
    input_ptr, weight_ptr, output_ptr,
    input_batch, input_channels, input_height, input_width,
    weight_channels, weight_height, weight_width,
    output_channels, output_height, output_width,
    conv_stride_h, conv_stride_w,
    conv_padding_h, conv_padding_w,
    conv_dilation_h, conv_dilation_w,
    pool_kernel_h, pool_kernel_w,
    pool_stride_h, pool_stride_w,
    pool_padding_h, pool_padding_w,
    groups: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    c_out = pid // (output_height * output_width)
    h_out = (pid % (output_height * output_width)) // output_width
    w_out = (pid % (output_height * output_width)) % output_width
    
    # Calculate input coordinates with padding
    h_in = h_out * pool_stride_h - pool_padding_h
    w_in = w_out * pool_stride_w - pool_padding_w
    h_in_start = max(0, h_in)
    h_in_end = min(input_height, h_in + pool_kernel_h)
    w_in_start = max(0, w_in)
    w_in_end = min(input_width, w_in + pool_kernel_w)
    
    # Handle case where padding is larger than input
    if h_in_start >= h_in_end or w_in_start >= w_in_end:
        # Fill with zeros if completely outside input region
        for b in range(input_batch):
            offset = output_ptr + ((b * output_channels + c_out) * output_height + h_out) * output_width + w_out
            tl.store(offset, 0.0)
        return
    
    # Get the correct weight channel group
    c_in_base = (c_out // (output_channels // groups)) * (input_channels // groups)
    
    # Initialize max value
    max_val = -float('inf')
    
    # Iterate over pool kernel and conv receptive field
    for h_pool in range(h_in_start, h_in_end):
        for w_pool in range(w_in_start, w_in_end):
            # Calculate conv input position
            h_conv_in = (h_pool * conv_stride_h - conv_padding_h) // conv_dilation_h
            w_conv_in = (w_pool * conv_stride_w - conv_padding_w) // conv_dilation_w
            
            if (0 <= h_conv_in < input_height and 0 <= w_conv_in < input_width):
                # Compute convolution sum over input channels
                conv_sum = 0.0
                for c_in_rel in range(input_channels // groups):
                    c_in = c_in_base + c_in_rel
                    for h_weight in range(weight_height):
                        for w_weight in range(weight_width):
                            h_input = h_conv_in + h_weight * conv_dilation_h
                            w_input = w_conv_in + w_weight * conv_dilation_w
                            
                            if (0 <= h_input < input_height and 0 <= w_input < input_width):
                                # Load input value
                                input_val = tl.load(input_ptr + ((0 * input_channels + c_in) * input_height + h_input) * input_width + w_input)
                                
                                # Load weight value
                                weight_val = tl.load(weight_ptr + ((c_out // (output_channels // groups)) * (input_channels // groups) + c_in_rel) * weight_height * weight_width + h_weight * weight_width + w_weight)
                                
                                conv_sum += input_val * weight_val
                
                if conv_sum > max_val:
                    max_val = conv_sum
    
    # Store the maxpool result
    for b in range(input_batch):
        offset = output_ptr + ((b * output_channels + c_out) * output_height + h_out) * output_width + w_out
        tl.store(offset, max_val if max_val != -float('inf') else 0.0)

@torch.fx.wrap
def fused_conv2d_maxpool_stride22(input_tensor, weight_tensor):
    # Get input dimensions
    input_batch, input_channels, input_height, input_width = input_tensor.shape
    output_channels, weight_channels, weight_height, weight_width = weight_tensor.shape
    
    # Calculate output dimensions for Conv2d with stride (2,2) and padding (3,3)
    conv_output_height = (input_height + 2 * 3 - 1 * (weight_height - 1) - 1) // 2 + 1
    conv_output_width = (input_width + 2 * 3 - 1 * (weight_width - 1) - 1) // 2 + 1
    
    # Calculate output dimensions for MaxPool2d with kernel 3 and stride 2
    output_height = (conv_output_height + 2 * 1 - 3) // 2 + 1
    output_width = (conv_output_width + 2 * 1 - 3) // 2 + 1
    
    # Create output tensor
    output = torch.empty((input_batch, output_channels, output_height, output_width), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Launch kernel
    total_elements = input_batch * output_channels * output_height * output_width
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    fused_conv2d_maxpool_kernel[(num_programs,)](
        input_tensor, weight_tensor, output,
        input_batch, input_channels, input_height, input_width,
        weight_channels, weight_height, weight_width,
        output_channels, output_height, output_width,
        2, 2,  # conv stride
        3, 3,  # conv padding
        1, 1,  # conv dilation
        3, 3,  # pool kernel
        2, 2,  # pool stride
        1, 1,  # pool padding
        1,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_conv2d_maxpool_stride22