import torch
import sys
import os

# Add pass_dir to sys.path so we can import shared_kernel
_pass_dir = os.path.dirname(os.path.abspath(__file__))
if _pass_dir not in sys.path:
    sys.path.insert(0, _pass_dir)
from shared_kernel import fused_silu_avgpool_flatten


def pattern(in_0):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return (tmp_3,)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return fused_silu_avgpool_flatten