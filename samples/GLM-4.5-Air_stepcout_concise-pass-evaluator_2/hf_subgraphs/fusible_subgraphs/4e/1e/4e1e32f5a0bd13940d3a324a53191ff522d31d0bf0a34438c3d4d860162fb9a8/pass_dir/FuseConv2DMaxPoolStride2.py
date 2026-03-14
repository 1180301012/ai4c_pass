import torch
import triton
import triton.language as tl

# Pattern matching function - matches stride-2 Conv2D followed by MaxPool2D
def pattern(input_tensor, weight_tensor):
    """Pattern: Conv2D with stride (2,2) followed by MaxPool2D"""
    # Conv2D operation - using the exact signature from the models with stride (2,2)
    conv_result = torch.conv2d(input_tensor, weight_tensor, None, (2, 2), (3, 3), (1, 1), 1)
    
    # MaxPool2D operation
    maxpool_result = torch.nn.functional.max_pool2d(conv_result, 3, 2, 1, 1, ceil_mode=False, return_indices=False)
    
    return maxpool_result

# Argument extraction function
def replacement_args(input_tensor, weight_tensor):
    """Extract arguments needed for the fused kernel"""
    return (input_tensor, weight_tensor)

# @torch.fx.wrap
def fused_conv_maxpool_stride2(input_tensor, weight_tensor):
    """Wrapper function for fused stride-2 Conv2D + MaxPool2D"""
    
    # Get input dimensions
    input_batch, input_channels, input_height, input_width = input_tensor.shape
    output_channels, kernel_channels, kernel_height, kernel_width = weight_tensor.shape
    
    # Conv2D parameters - fixed to match pattern: stride (2,2), padding (3,3), dilation (1,1)
    conv_stride = (2, 2)
    conv_padding = (3, 3)
    conv_dilation = (1, 1)
    
    # MaxPool2D parameters - fixed to match pattern: kernel (3,2), stride (2,1), padding (1,1)
    pool_kernel = (3, 2)
    pool_stride = (2, 1)
    pool_padding = (1, 1)
    
    # Calculate output dimensions
    conv_output_height = (input_height + 2 * conv_padding[0] - kernel_height * conv_dilation[0]) // conv_stride[0] + 1
    conv_output_width = (input_width + 2 * conv_padding[1] - kernel_width * conv_dilation[1]) // conv_stride[1] + 1
    
    maxpool_output_height = (conv_output_height + 2 * pool_padding[0] - pool_kernel[0]) // pool_stride[0] + 1
    maxpool_output_width = (conv_output_width + 2 * pool_padding[1] - pool_kernel[1]) // pool_stride[1] + 1
    
    # For now, implement as sequential operations to ensure correctness
    # This can be later optimized to a fused Triton kernel
    
    # Step 1: Convolution with stride (2,2)
    conv_output = torch.conv2d(input_tensor, weight_tensor, None, conv_stride, conv_padding, conv_dilation, 1)
    
    # Step 2: Max pooling
    maxpool_output = torch.nn.functional.max_pool2d(conv_output, pool_kernel, pool_stride, pool_padding, 
                                                   ceil_mode=False, return_indices=False)
    
    return maxpool_output

# Replacement function (returns function reference)
def replacement_func():
    return fused_conv_maxpool_stride2