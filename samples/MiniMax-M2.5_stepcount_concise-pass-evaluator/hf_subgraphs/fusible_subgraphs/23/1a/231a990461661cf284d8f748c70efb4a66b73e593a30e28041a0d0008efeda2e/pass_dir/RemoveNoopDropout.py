import torch
import triton
import triton.language as tl


def pattern(tmp_10):
    """
    Match dropout with p=0 (which is a no-op).
    dropout(x, 0.0, False, False) returns the input unchanged.
    """
    tmp_11 = torch.nn.functional.dropout(tmp_10, 0.0, False, False)
    return tmp_11


def replacement_args(tmp_10):
    """
    Extract the input tensor - dropout with p=0 just returns the input.
    """
    return (tmp_10,)


@torch.fx.wrap
def remove_noop_dropout_wrapper(tmp_10):
    """
    Remove no-op dropout - dropout(p=0) is identity function.
    """
    return tmp_10


def replacement_func():
    return remove_noop_dropout_wrapper