import torch
import triton
import triton.language as tl
from pass_dir.shared_kernels import dispatch_wrapper


def pattern(x):
    pool_out = torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)
    return pool_out


def replacement_args(x):
    return (x,)


def replacement_func():
    return dispatch_wrapper