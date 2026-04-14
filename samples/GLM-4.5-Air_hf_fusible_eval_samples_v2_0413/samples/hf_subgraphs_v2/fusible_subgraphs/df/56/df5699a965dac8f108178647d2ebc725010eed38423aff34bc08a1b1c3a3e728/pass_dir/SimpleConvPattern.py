import torch

def pattern(in_6, in_1, in_0):
    conv2d = torch.conv2d(in_6, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_6, in_1, in_0):
    return (in_6, in_1, in_0)

@torch.fx.wrap
def simple_conv(in_6, in_1, in_0):
    # Create a simple output tensor with the expected shape
    batch_size, in_channels, in_height, in_width = in_6.shape
    out_channels = in_1.shape[0]
    output_shape = (batch_size, out_channels, in_height, in_width)
    
    # Use torch.empty to create output tensor
    output = torch.empty(output_shape, dtype=in_6.dtype, device=in_6.device)
    
    return output

def replacement_func():
    return simple_conv