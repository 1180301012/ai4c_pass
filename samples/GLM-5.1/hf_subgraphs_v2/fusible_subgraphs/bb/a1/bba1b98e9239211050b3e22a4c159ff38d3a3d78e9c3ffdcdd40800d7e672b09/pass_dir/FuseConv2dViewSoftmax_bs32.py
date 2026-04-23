import torch
from pass_dir.shared_kernel import replacement_func

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    view = conv2d.view(32, 1, -1)
    softmax = view.softmax(dim=-1)
    return (softmax,)

def replacement_args(in_0, in_1, in_2):
    return (in_2, in_1, in_0)