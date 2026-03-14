import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match dropout with p=0.0 (training=False), which is a no-op.
    """
    # dropout with p=0.0 and training=False is a no-op
    result = torch.nn.functional.dropout(x, 0.0, False, False)
    return result


def replacement_args(x):
    return (x,)


def replacement_func():
    # Identity function - dropout with p=0.0 is a no-op
    def identity(x):
        return x
    return identity