import torch
from pass_dir.shared_kernel import dispatch_fused_div_transpose


def pattern(x):
    # Try aten.div.Scalar for the _decomposed graph with Python-float constant
    tmp_0 = torch.ops.aten.div.Scalar(x, 2.8284271247461903)
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(x):
    return (x, "r28284")


def replacement_func():
    return dispatch_fused_div_transpose