import torch
import triton
import triton.language as tl


def pattern(tmp_1):
    """
    Eliminate dropout with p=0.0 (no-op).
    When p=0.0, dropout returns the input unchanged.
    """
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return tmp_2


def replacement_args(tmp_1):
    return (tmp_1,)


def replacement_func():
    # Identity function - dropout with p=0.0 does nothing
    return torch.nn.functional.identity