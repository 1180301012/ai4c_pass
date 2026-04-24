"""
Pass: FuseLinearAndScaleIndependent  (routing: "mul")
Matches elementwise scale multiply across all models (single-tensor output).

For rtmpose:  tmp_3 = in_2 * in_1   shape [..., 256]
The dispatch route "mul" runs a Triton broadcast-multiply kernel.

All passes share the same replacement_func (dispatch) to avoid
output_pass_replacement_func_limit dropping any pass.
"""
import torch
import triton
import triton.language as tl
from pass_dir.dispatch import dispatch


def pattern(in_1, in_2):
    return in_2 * in_1


def replacement_args(in_1, in_2):
    return (in_1, in_2, "mul")


def replacement_func():
    return dispatch