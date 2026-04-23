import torch
import operator
from pass_dir.matmul_view_kernel import matmul_fused

# Pattern: matches `in_1 @ in_0` which maps to operator.matmul in FX
def pattern(in_0, in_1):
    matmul = operator.matmul(in_1, in_0)
    return matmul

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return matmul_fused