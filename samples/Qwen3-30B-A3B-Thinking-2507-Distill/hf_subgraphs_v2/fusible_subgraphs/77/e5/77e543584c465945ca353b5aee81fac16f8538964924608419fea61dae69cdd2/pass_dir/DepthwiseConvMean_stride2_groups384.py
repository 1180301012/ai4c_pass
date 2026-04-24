"""Pass: Depthwise conv2d (stride=2, groups=384) → single tensor."""
import torch
from pass_dir.shared_dw_conv_mean import dispatch_dw_conv
def pattern(in_1, in_0):
    return torch.conv2d(in_1, in_0, None, (2, 2), (1, 1), (1, 1), 384)
def replacement_args(in_1, in_0):
    return (in_1, in_0, "s2_c384")
def replacement_func():
    return dispatch_dw_conv