import torch
import triton
import triton.language as tl

from pass_dir.shared_kernel import fused_conv_post


# ── Pattern: returns ONLY tmp_8 (single output) – same as FuseConvPostDrop005
# This variant is kept for fallback coverage and satisfies the
# replacement_func_limit by sharing fused_conv_post with pass 005.
def pattern(x_conv, in3):
    gelu_x = torch.nn.functional.gelu(x_conv[slice(None, None, None), slice(None, None, None), slice(None, -1, None)])
    t      = gelu_x.transpose(1, 2)
    s      = in3 + t
    return s          # returns tmp_8


def replacement_args(x_conv, in3):
    return (x_conv, in3, 'route_sum')   # pass route tag for replacement_func_limit


def replacement_func():
    return fused_conv_post