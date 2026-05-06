import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper  # noqa: F401


def pattern(in_0, in_1, in_3):
    """
    Conv2d(B=64) + view:  bfloat16/8 AND float32/8 graphs
    in_0=bias[256], in_1=weight[256,512,1,1], in_3=input[64,512,64,64]
    FX graph: torch.conv2d(...).view(64, 256, -1)
    """
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d.view(64, 256, -1)     # [64, 256, 4096]
    return tmp_3


def replacement_args(in_0, in_1, in_3):
    return (in_0, in_1, in_3, "conv_view")


def replacement_func():
    return dispatch_wrapper