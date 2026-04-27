import torch
import triton
import triton.language as tl
from pass_dir.shared_ln import fused_dispatch


# Match only the layer_norm call (single-tensor output).
# in_0 = bias, in_1 = weight, x = input (already reshaped tmp_3)
def pattern(in_0, in_1, x):
    return torch.nn.functional.layer_norm(x, (16,), in_1, in_0, 1e-05)


def replacement_args(in_0, in_1, x):
    return (in_0, in_1, x, "route_16")


def replacement_func():
    return fused_dispatch