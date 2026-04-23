import torch
from pass_dir.shared_kernels import fused_dispatch_wrapper

# Pattern matching for linear + permute(0, 2, 1)
# This is shape-independent and should match all graphs
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    permuted = linear.permute(0, 2, 1)
    return (permuted,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "linear_permute")

def replacement_func():
    return fused_dispatch_wrapper