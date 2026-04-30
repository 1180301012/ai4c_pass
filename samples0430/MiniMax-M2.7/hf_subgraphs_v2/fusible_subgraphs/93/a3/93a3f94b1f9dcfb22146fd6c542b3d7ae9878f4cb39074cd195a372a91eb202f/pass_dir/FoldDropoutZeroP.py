import torch
import triton
import triton.language as tl


def pattern(x):
    """Pattern to match: dropout with p=0.0 -> should be simplified to identity"""
    return torch.nn.functional.dropout(x, p=0.0, training=False)


def replacement_args(x):
    return (x,)


# Module-level function
def fold_dropout(x):
    return x


def replacement_func():
    return fold_dropout