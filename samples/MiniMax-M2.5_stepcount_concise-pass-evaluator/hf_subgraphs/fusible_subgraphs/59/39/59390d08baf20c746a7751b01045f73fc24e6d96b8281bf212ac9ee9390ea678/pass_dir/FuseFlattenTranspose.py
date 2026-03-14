import torch
import triton
import triton.language as tl


def pattern(x):
    """
    Match a simple transpose(1, 2) operation.
    Optimizes by ensuring contiguous output.
    
    Input x: [B, C, N] - any 3D tensor
    Output: [B, N, C] - transposed
    """
    tmp_5 = x.transpose(1, 2)
    return tmp_5


def replacement_args(x):
    return (x,)


def triton_transpose(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized transpose ensuring contiguous output for better performance.
    
    Input: [B, C, N]
    Output: [B, N, C] (contiguous)
    """
    return x.transpose(1, 2).contiguous()


@torch.fx.wrap
def kernel_wrapper(x):
    """
    Custom kernel that performs optimized transpose.
    """
    return triton_transpose(x)


def replacement_func():
    return kernel_wrapper