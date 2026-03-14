import torch
import triton
import triton.language as tl

def pattern(in_4, in_0, in_1, in_3, in_2):
    """Match just batch_norm to test"""
    tmp_0 = in_0
    tmp_1 = in_1
    tmp_2 = in_2
    tmp_3 = in_3
    tmp_4 = torch.nn.functional.batch_norm(in_4, tmp_0, tmp_1, tmp_3, tmp_2, False, 0.1, 1e-05)
    return tmp_4

def replacement_args(in_4, in_0, in_1, in_3, in_2):
    return (in_4, in_0, in_1, in_3, in_2)

def replacement_func():
    return torch.nn.functional.batch_norm