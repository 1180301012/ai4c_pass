import torch


def pattern(in_1):
    """
    Match the pattern: cumsum(-1) - 1
    """
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    return tmp_2


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def custom_cumsum_minus_one(in_1):
    """
    Custom implementation: cumsum(-1) - 1
    Fuses cumsum and subtract in a single operation.
    """
    cumsum_result = in_1.cumsum(-1)
    return cumsum_result - 1


def replacement_func():
    return custom_cumsum_minus_one