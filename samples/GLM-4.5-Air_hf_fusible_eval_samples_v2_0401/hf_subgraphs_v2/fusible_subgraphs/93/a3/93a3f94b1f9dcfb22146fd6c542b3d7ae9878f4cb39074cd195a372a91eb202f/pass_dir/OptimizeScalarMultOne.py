import torch
import triton
import triton.language as tl

def pattern(matmul):
    """
    Pattern matching the scalar multiplication by 1.0:
    multiplication by 1.0 is the identity function and can be removed
    """
    tmp_1 = matmul * 1.0
    return tmp_1

def replacement_args(matmul):
    return (matmul,)

@torch.fx.wrap
def remove_scalar_mult_one(matmul):
    """
    Remove scalar multiplication by 1.0 - this is the identity function
    """
    return matmul

def replacement_func():
    return remove_scalar_mult_one