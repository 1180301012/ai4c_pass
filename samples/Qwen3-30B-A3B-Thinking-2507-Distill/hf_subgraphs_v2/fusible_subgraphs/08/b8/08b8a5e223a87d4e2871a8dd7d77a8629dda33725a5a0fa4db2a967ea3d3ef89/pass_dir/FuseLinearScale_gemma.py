"""
Pass: FuseLinearScale_gemma  (routing: "linear")
Matches torch.nn.functional.linear(x, weight, None) across ALL models.

Gemma/SmolLM3/rtmpose all use:  linear = linear(in_3, in_0, None)
The fused SmolLM3 pass runs FIRST, so this pass sees only un-fused linear ops.

All passes share the same replacement_func (dispatch) to avoid
output_pass_replacement_func_limit dropping any pass.
"""
import torch
import triton
import triton.language as tl
from pass_dir.dispatch import dispatch


def pattern(in_0, in_3):
    return torch.nn.functional.linear(in_3, in_0, None)


def replacement_args(in_0, in_3):
    return (in_0, in_3, "linear")


def replacement_func():
    return dispatch