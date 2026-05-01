import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fused_permute_reshape_sigmoid


def pattern(conv_output):
    t = conv_output.permute(0, 2, 3, 1)
    r = t.reshape(12, -1, 9)
    s = torch.nn.functional.sigmoid(r)
    return s


def replacement_args(conv_output):
    return (conv_output,)


def replacement_func():
    return fused_permute_reshape_sigmoid