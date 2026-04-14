"""
RTMPose broadcast-multiply pass: matches the standalone in_2 * in_1 operation
where in_1 is a 1-D scale vector.  Both outputs of the RTMPose model are
optimized independently: this pass handles the multiply; LinearGEMM_RTMPose
handles the linear.

Uses the shared dispatch_all function (route="rtmpose_mul").
"""

import torch
from pass_dir.shared_kernels import dispatch_all


def pattern(in_1, in_2):
    tmp_3 = in_2 * in_1
    return tmp_3


def replacement_args(in_1, in_2):
    # a0=scale(in_1), a1=tensor(in_2), a2=dummy(in_1 reused), route="rtmpose_mul"
    return (in_1, in_2, in_1, "rtmpose_mul")


def replacement_func():
    return dispatch_all