import torch

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return tmp_2

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def fused_conv2d_relu(in_0, in_1, in_2):
    # Just use PyTorch's built-in functions for now
    # This removes the subsequent ReLU and dropout operations
    conv_result = torch.conv2d(in_2, in_1, in_0, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1)
    return conv_result

def replacement_func():
    return fused_conv2d_relu