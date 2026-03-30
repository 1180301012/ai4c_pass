import torch

# Very basic pattern - just conv2d
def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    def basic_conv(in_0, in_1):
        return torch.ones((1, 640, 16, 16))  # dummy output with same shape
    
    return basic_conv