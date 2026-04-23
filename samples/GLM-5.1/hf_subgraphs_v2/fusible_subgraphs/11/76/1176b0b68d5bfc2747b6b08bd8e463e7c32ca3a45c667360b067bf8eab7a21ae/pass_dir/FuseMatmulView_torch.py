import torch
from pass_dir.matmul_view_kernel import matmul_fused

# Pattern: matches `torch.matmul(in_1, in_0)` which maps to torch.matmul in FX
def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul

def replacement_args(in_0, in_1):
    return (in_0, in_1)

def replacement_func():
    return matmul_fused