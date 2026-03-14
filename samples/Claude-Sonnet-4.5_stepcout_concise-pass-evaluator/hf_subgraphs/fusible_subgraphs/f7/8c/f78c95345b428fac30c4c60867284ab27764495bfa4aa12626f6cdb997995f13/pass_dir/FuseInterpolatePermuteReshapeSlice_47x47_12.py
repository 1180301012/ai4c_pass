import torch


def pattern(in_1):
    """
    Pattern for permute + reshape: [1, 12, 47, 47] -> [2209, 12]
    """
    tmp_2 = in_1.permute(0, 2, 3, 1)
    tmp_3 = tmp_2.reshape(2209, -1)
    return tmp_3


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def optimized_permute_reshape_47x47(in_1):
    """
    Native PyTorch is already highly optimized for these small operations.
    Just ensure contiguous memory layout for best performance.
    """
    return in_1.permute(0, 2, 3, 1).reshape(2209, -1)


def replacement_func():
    return optimized_permute_reshape_47x47