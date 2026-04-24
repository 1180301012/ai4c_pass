"""Pass: Depthwise conv2d (stride=1, groups=256) → single tensor."""
import torch
from pass_dir.shared_dw_conv_mean import dispatch_dw_conv
def pattern(in_1, in_0):
    return torch.conv2d(in_1, in_0, None, (1, 1), (1, 1), (1, 1), 256)
def replacement_args(in_1, in_0):
    return (in_1, in_0, "s1_c256")
def replacement_func():
    return dispatch_dw_conv