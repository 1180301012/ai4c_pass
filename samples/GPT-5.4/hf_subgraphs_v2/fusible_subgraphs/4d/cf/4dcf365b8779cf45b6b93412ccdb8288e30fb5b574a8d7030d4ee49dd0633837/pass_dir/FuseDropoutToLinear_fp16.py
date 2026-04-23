import torch
import triton
import triton.language as tl
from pass_dir.shared_linear_kernels import shared_linear_dispatch


# Pattern matching function

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.dropout(in_2, p=0.0, training=False)
    to = tmp_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return (linear,)


# Argument extraction function

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, 'dropout_to_linear_fp16')


@triton.jit
def _marker_kernel(x_ptr):
    pass


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return shared_linear_dispatch