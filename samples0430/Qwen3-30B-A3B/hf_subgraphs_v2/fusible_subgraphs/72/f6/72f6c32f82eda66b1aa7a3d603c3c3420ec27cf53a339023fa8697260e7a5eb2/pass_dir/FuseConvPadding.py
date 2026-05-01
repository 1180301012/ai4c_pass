import torch

def pattern(in_1, in_0):
    conv = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    pad = torch.nn.functional.pad(conv, [2, 2, 2, 2], 'constant', None)
    return pad

def replacement_args(in_1, in_0):
    return (in_1, in_0)

def replacement_func():
    def fuse_conv_padding(in_1, in_0):
        return torch.conv2d(in_1, in_0, None, (1, 1), (2, 2), (1, 1), 1)
    return fuse_conv_padding