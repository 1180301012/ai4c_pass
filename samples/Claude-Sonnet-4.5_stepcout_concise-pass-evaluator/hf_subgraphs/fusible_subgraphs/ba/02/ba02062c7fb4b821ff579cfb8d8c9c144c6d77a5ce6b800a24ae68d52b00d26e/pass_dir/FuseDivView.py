import torch


def pattern(in_1):
    """
    Pattern to match: division of in_1 by 5.656854249492381
    """
    result = in_1 / 5.656854249492381
    return result


def replacement_args(in_1):
    return (in_1,)


@torch.fx.wrap
def optimized_div(in_1):
    """
    Optimized implementation using multiplication by reciprocal
    PyTorch's mul is typically faster than div
    """
    # Precompute reciprocal: 1 / 5.656854249492381
    reciprocal = 0.17677669529663687
    return in_1 * reciprocal


def replacement_func():
    return optimized_div