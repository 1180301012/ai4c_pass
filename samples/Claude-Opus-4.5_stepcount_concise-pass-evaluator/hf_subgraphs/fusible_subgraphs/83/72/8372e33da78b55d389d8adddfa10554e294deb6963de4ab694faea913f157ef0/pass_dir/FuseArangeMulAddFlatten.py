import torch
import triton
import triton.language as tl


def pattern(mul_result, in_0):
    """
    Pattern to match the view + unsqueeze + add + flatten computation.
    This matches the operations after arange * num_segments.
    """
    tmp_4 = mul_result.view((1,))
    tmp_5 = tmp_4.unsqueeze(-1)
    tmp_6 = tmp_5 + in_0
    tmp_7 = tmp_6.view(-1)
    return tmp_7


def replacement_args(mul_result, in_0):
    return (mul_result, in_0)


@torch.fx.wrap
def optimized_view_unsqueeze_add_flatten(mul_result, in_0):
    """
    Optimized implementation - add first (broadcasts), then flatten.
    This preserves the original memory layout during the add operation.
    """
    # Add with broadcast (mul_result broadcasts to in_0's shape), then flatten
    return (in_0 + mul_result).view(-1)


def replacement_func():
    return optimized_view_unsqueeze_add_flatten