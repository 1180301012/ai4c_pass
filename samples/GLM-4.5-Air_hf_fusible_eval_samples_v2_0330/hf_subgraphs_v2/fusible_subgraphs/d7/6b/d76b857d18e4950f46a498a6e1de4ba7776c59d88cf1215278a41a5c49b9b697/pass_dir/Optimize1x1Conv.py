import torch

def pattern(input, weight, bias):
    """Match conv2d operation with specific parameters for 1x1 convolution"""
    result = torch.conv2d(input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return result

def replacement_args(input, weight, bias):
    """Extract arguments for the replacement"""
    return (input, weight, bias)

@torch.fx.wrap
def optimized_1x1_conv(x, weight, bias):
    """Optimized 1x1 convolution using matrix multiplication approach"""
    # Get dimensions
    batch_size, in_channels, height, width = x.shape
    out_channels = weight.shape[0]
    
    # Reshape input for matrix multiplication: [batch, height*width, channels]
    x_reshaped = x.reshape(batch_size, height * width, in_channels)
    
    # Reshape weight for matrix multiplication: [out_channels, in_channels]
    weight_reshaped = weight.reshape(out_channels, in_channels)
    
    # Perform matrix multiplication using PyTorch's optimized matmul
    # result shape: [batch, height*width, out_channels]
    conv_result = torch.matmul(x_reshaped, weight_reshaped.transpose(0, 1))
    
    if bias is not None:
        # Add bias: [out_channels] -> [1, out_channels, 1] -> [batch, out_channels, height*width]
        bias_reshaped = bias.reshape(1, out_channels, 1)
        conv_result = conv_result + bias_reshaped
    
    # Reshape back to original format: [batch, out_channels, height, width]
    return conv_result.reshape(batch_size, out_channels, height, width)

def replacement_func():
    """Return the optimized function"""
    return optimized_1x1_conv