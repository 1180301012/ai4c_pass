import torch


def pattern(in_1):
    """
    Pattern to match: division of in_1 by 2.0
    """
    result = in_1 / 2.0
    return result


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def optimized_div_2(in_1):
    """
    Optimized implementation using multiplication by reciprocal
    PyTorch's mul is typically faster than div
    """
    # Precompute reciprocal: 1 / 2.0 = 0.5
    reciprocal = 0.5
    return in_1 * reciprocal


def replacement_func():
    return optimized_div_2