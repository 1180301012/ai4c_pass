import torch
from pass_dir.shared_kernels import fused_dispatch_wrapper

# Pattern matching for batch size 12
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.permute(0, 2, 1)
    tmp_4 = tmp_3.reshape(12, -1, 16, 16)
    tmp_5 = torch.nn.functional.interpolate(tmp_4, size=(128, 128), mode='bilinear', align_corners=False)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "bs12")

def replacement_func():
    return fused_dispatch_wrapper