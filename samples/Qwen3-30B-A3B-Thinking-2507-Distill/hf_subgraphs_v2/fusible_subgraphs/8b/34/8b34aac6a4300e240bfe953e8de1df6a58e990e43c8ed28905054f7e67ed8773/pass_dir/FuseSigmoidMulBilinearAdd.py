"""
Pass: FuseSigmoidMulBilinearAdd

Pattern: sigmoid(conv2d_out) * in_2  (matches the 16x16 branch)

Fuses torch.sigmoid + element-wise multiply into a single Triton kernel.
Uses the shared dispatch_sigmoid_mul from pass_dir.shared_kernel so that
replacement_func() returns the SAME Python object as FuseBilinearSigmoidMul,
avoiding the output_pass_replacement_func_limit.
"""

import torch
from pass_dir.shared_kernel import dispatch_sigmoid_mul


# ---------------------------------------------------------------------------
# Pattern / replacement hooks used by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(conv2d_out, in_2):
    tmp_6 = torch.sigmoid(conv2d_out)
    tmp_7 = in_2 * tmp_6
    return tmp_7


def replacement_args(conv2d_out, in_2):
    return (conv2d_out, in_2)


def replacement_func():
    return dispatch_sigmoid_mul