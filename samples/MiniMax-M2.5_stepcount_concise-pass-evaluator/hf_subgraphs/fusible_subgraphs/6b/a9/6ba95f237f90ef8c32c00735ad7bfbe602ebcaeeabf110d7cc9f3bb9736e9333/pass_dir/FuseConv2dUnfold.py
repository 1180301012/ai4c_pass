import torch
import triton
import triton.language as tl


def pattern(tmp_2):
    """
    Match reshape after unfold.
    Input: (1, 512, 256) -> Output: (1, 128, 4, 256)
    """
    tmp_3 = tmp_2.reshape(1, 128, 4, -1)
    return tmp_3


def replacement_args(tmp_2):
    return (tmp_2,)


@torch.fx.wrap
def triton_reshape(input_tensor):
    """Use PyTorch's optimized reshape (view)."""
    return input_tensor.reshape(1, 128, 4, -1)


def replacement_func():
    return triton_reshape