import torch
import triton
import triton.language as tl
from pass_dir.shared_relu_dispatch import relu_fuse_dispatch


# Variant C: aten.relu.default (non-inplace) + aten.view.default([-1, 1280])
def pattern(in_0):
    tmp_0 = torch.ops.aten.relu.default(in_0)
    tmp_1 = torch.ops.aten.view.default(tmp_0, [-1, 1280])
    return tmp_1


def replacement_args(in_0):
    return (in_0, "route_c")


def replacement_func():
    return relu_fuse_dispatch