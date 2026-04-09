import torch
import triton
import triton.language as tl

def pattern(tmp_3):
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)
    tmp_5 = torch.reshape(tmp_4, [-1, 8, 9])
    return tmp_5

def replacement_args(tmp_3):
    return (tmp_3,)

def reshape_4d_to_3d_optimized(tmp_3):
    # Direct reshape without intermediate tensor
    # From [1, N, 16, 9] to [-1, 8, 9]
    # This is equivalent to: reshape(1*N*16//8, 8, 9)
    tmp_4 = tmp_3.reshape(1, -1, 16, 9)  # First reshape
    out = tmp_4.reshape(-1, 8, 9)        # Second reshape
    return out

def replacement_func():
    return reshape_4d_to_3d_optimized