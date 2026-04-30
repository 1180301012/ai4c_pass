import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(x, running_mean, running_var, weight, bias):
    """
    Match batch_norm inference only.
    x: [N, C]  (spatial-mean result, output of dropout which is identity here)
    Returns tmp_8 [N, C].
    """
    out = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    return out


def replacement_args(x, running_mean, running_var, weight, bias):
    # Route "bn_inference": a=x, b=running_mean, c=running_var, d=weight, e=bias
    return (x, running_mean, running_var, weight, bias, "bn_inference")


def replacement_func():
    return dispatch_wrapper