import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_ops import universal_fused_op


def pattern(x):
    tmp_3 = torch.nn.functional.gelu(x)
    tmp_4 = torch.nn.functional.dropout(tmp_3, 0.0, False, False)
    return tmp_4


def replacement_args(x):
    # ctx1 and ctx2 are dummy (same as x, unused in gelu_default route)
    return (x, x, x, 'gelu_default')


def replacement_func():
    return universal_fused_op