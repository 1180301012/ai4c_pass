import torch

def pattern(in_2, in_1, in_0):
    """Simple pattern to test matching"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

def replacement_func():
    @torch.fx.wrap
    def identity_conv(in_2, in_1, in_0):
        return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    
    return identity_conv