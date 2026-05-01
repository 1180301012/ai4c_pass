import torch
from pass_dir.shared_fused_ln import fused_gelu_add_ln_dispatch


# Pass B: fuse layer_norm (C=128) using a Triton kernel.
# The final view (tmp_12 = tmp_11.view(1,16,12,128)) stays outside this pattern.
def pattern(in_0, in_1, x):
    return torch.nn.functional.layer_norm(x, (128,), in_1, in_0, 1e-06)


def replacement_args(in_0, in_1, x):
    # dispatch(arg0=in_0=bias, arg1=in_1=weight, arg2=x) → layer_norm branch
    return (in_0, in_1, x)


def replacement_func():
    return fused_gelu_add_ln_dispatch