import torch


def pattern(in_3, in_0):
    """
    Simple pattern: multiply in_3 * in_0
    This pattern matches element-wise multiplications in the graph.
    """
    tmp = in_3 * in_0
    return tmp


def replacement_args(in_3, in_0):
    return (in_3, in_0)


@torch.fx.wrap
def optimized_mul(in_3, in_0):
    """
    Optimized multiplication - direct call with minimal overhead.
    """
    return in_3 * in_0


def replacement_func():
    return optimized_mul