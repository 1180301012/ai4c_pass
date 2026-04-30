import torch
import triton
import triton.language as tl
from pass_dir.shared_relu6_gap import dispatch_fused_relu6_gap_avgpool


def pattern(x):
    t0 = torch.nn.functional.hardtanh(x, 0.0, 6.0, True)
    t1 = torch.nn.functional.adaptive_avg_pool2d(t0, (1, 1))
    t2 = t1.view(64, -1)
    t3 = torch.flatten(t2, 1)
    return t3


def replacement_args(x):
    return (x,)


def replacement_func():
    return dispatch_fused_relu6_gap_avgpool