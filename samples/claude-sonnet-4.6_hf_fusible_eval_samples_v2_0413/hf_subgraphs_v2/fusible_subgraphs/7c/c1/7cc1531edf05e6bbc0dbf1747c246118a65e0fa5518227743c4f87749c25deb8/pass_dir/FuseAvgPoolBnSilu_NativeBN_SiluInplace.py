import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_avgpool_bn_silu_kernel import fused_avgpool_bn_silu

import torch
import operator


# Pattern: reshape → avg_pool2d → _native_batch_norm_legit_no_training → getitem → silu_ (inplace)

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = torch.ops.aten.reshape.default(in_4, [1, 512, 16, 16])
    tmp_5 = torch.ops.aten.avg_pool2d.default(tmp_4, [2, 2], [2, 2], [0, 0], False, True, None)
    bn_out = torch.ops.aten._native_batch_norm_legit_no_training.default(tmp_5, in_3, in_2, in_0, in_1, 0.1, 1e-05)
    tmp_6 = operator.getitem(bn_out, 0)
    tmp_7 = torch.ops.aten.silu_.default(tmp_6)
    return tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_avgpool_bn_silu