import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shared_avgpool_bn_silu_kernel import fused_avgpool_bn

import torch


# Pattern: reshape + avg_pool2d + batch_norm  (no silu)
# ForceArgsTracer normalises F.silu kwargs→positional, breaking the match.
# Omitting silu from the pattern lets silu execute separately (correct).
# The fused kernel covers: reshape → 2×2 avg-pool → batch-norm.

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_4 = in_4.reshape(1, 512, 16, 16)
    tmp_5 = torch.nn.functional.avg_pool2d(tmp_4, 2, 2, 0, False, True, None)
    tmp_6 = torch.nn.functional.batch_norm(tmp_5, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return fused_avgpool_bn