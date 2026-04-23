"""
Pass for fusing element-wise add with dropout2d.

Pattern: tmp_3 = in_4 + in_3 followed by dropout2d(tmp_3, 0.1, False, False)

Since train=False in the original pattern, dropout is essentially identity,
so the fusion reduces to a single optimized add kernel.
"""

import torch
from pass_dir.SharedKernelDispatch import dispatch_kernel


def pattern(in_3, in_4):
    """
    Match the add + dropout2d pattern from the model.
    
    Pattern:
        tmp_3 = in_4 + in_3
        tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
        return tmp_4
    """
    tmp_3 = in_4 + in_3
    tmp_4 = torch.nn.functional.dropout2d(tmp_3, 0.1, False, False)
    return tmp_4


def replacement_args(in_3, in_4):
    """
    Extract arguments for the replacement function.
    
    For add_dropout route:
    - in_2 = in_3 (first operand of add)
    - in_1 = in_4 (second operand of add)
    - in_0 = None (unused)
    - stride_h = 1 (unused)
    - stride_w = 1 (unused)
    - route = "add_dropout"
    """
    return (in_3, in_4, None, 1, 1, "add_dropout")


def replacement_func():
    """
    Returns the shared dispatch wrapper function.
    The dispatch will route to the add_dropout kernel based on route string.
    """
    return dispatch_kernel