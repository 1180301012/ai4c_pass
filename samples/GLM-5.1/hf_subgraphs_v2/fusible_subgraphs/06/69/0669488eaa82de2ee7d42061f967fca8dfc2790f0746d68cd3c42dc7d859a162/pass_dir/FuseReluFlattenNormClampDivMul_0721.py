import torch
import triton
import triton.language as tl

from pass_dir._fused_relu_norm_kernel import fused_relu_norm_dispatch


# Pattern matching function - must mirror model.py exactly
# This matches the computation: relu -> flatten -> norm -> *0.07216878364870322 -> clamp -> div -> *in_0
def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace = True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim = -1, keepdim = True)
    tmp_4 = tmp_3 * 0.07216878364870322
    tmp_5 = tmp_4.clamp(min = 1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_0721")


def replacement_func():
    return fused_relu_norm_dispatch