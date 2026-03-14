import torch
import triton
import triton.language as tl

def pattern(input_tensor, weight, bias):
    """Match conv2d (1x1) operation"""
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    return conv_out

def replacement_args(input_tensor, weight, bias):
    return (input_tensor, weight, bias)

@torch.fx.wrap
def fast_conv2d_1x1(input_tensor, weight, bias):
    """Use native PyTorch F.linear for 1x1 conv"""
    import torch.nn.functional as F
    batch_size, in_channels, H, W = input_tensor.shape
    out_channels = weight.shape[0]
    
    # Reshape for linear operation
    x = input_tensor.permute(0, 2, 3, 1).reshape(-1, in_channels)
    w = weight.reshape(out_channels, in_channels)
    
    # Use F.linear which is optimized
    out = F.linear(x, w, bias)
    
    # Reshape back
    return out.reshape(batch_size, H, W, out_channels).permute(0, 3, 1, 2)

def replacement_func():
    return fast_conv2d_1x1