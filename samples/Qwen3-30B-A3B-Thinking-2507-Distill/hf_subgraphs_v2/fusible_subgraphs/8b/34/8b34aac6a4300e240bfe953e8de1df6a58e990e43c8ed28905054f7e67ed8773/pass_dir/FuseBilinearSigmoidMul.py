"""
Pass: FuseBilinearSigmoidMul

Pattern: sigmoid(x) * y  (matches the 64x64 branch: sigmoid(interpolated_in4) * in_3)

Fuses torch.sigmoid + element-wise multiply into a single Triton kernel.
Uses the shared dispatch_sigmoid_mul from pass_dir.shared_kernel so that
replacement_func() returns the SAME Python object as FuseSigmoidMulBilinearAdd,
avoiding the output_pass_replacement_func_limit.
"""

import torch
from pass_dir.shared_kernel import dispatch_sigmoid_mul


# ---------------------------------------------------------------------------
# Pattern / replacement hooks used by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(in_4, in_3):
    tmp_4 = torch.sigmoid(in_4)
    tmp_5 = in_3 * tmp_4
    return tmp_5


def replacement_args(in_4, in_3):
    return (in_4, in_3)


def replacement_func():
    return dispatch_sigmoid_mul