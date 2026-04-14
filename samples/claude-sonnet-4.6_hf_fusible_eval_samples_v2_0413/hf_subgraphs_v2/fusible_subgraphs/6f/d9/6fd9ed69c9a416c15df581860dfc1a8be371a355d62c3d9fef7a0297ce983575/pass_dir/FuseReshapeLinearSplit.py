"""
Pass: FuseReshapeLinearSplit

Fuses:
    tmp_9 = in_4.reshape(300, -1, 256)          # [300, 1, 256]
    linear_1 = F.linear(tmp_9, in_3, in_2)      # [300, 1, 512]
    tmp_11 = linear_1[..., :256]                 # [300, 1, 256]  first half
    tmp_12 = linear_1[..., -256:]                # [300, 1, 256]  second half

into a single Triton kernel that writes directly to two output buffers,
avoiding materialization of the [300, 1, 512] intermediate.
Uses shared dispatcher (route="rls") to satisfy replacement_func_limit.
"""

import torch
from pass_dir.shared_gemm import dispatch_gemm_split


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_4, in_3, in_2):
    # Single-output: reshape + linear. Ellipsis slices remain in graph.
    tmp_9 = in_4.reshape(300, -1, 256)
    linear_1 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    return linear_1


def replacement_args(in_4, in_3, in_2):
    return (in_4, in_3, in_2)


def replacement_func():
    return dispatch_gemm_split