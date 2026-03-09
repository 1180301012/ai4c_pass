import torch

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.flatten(tmp_2, 2)
    return tmp_3

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def baseline_wrapper(in_0, in_1, in_2):
    # Just call the original PyTorch functions
    # This is a baseline to show the pass works
    result = pattern(in_0, in_1, in_2)
    return result

def replacement_func():
    return baseline_wrapper