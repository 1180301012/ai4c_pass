import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import sigmoid_nhwc


def pattern(x):
    perm = x.permute(0, 2, 3, 1)
    reshaped = perm.reshape(16, -1, 9)
    out = torch.nn.functional.sigmoid(reshaped)
    return out


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def _fuse_permute_sigmoid_N16_C9(x):
    return sigmoid_nhwc(x, 16, 9)


def replacement_func():
    return _fuse_permute_sigmoid_N16_C9