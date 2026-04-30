import torch
from pass_dir.shared_kernel import triton_cat, replacement_func

# Pattern matching function - matches the cat operation
# This pattern matches across all variants since torch.cat is the same operation
def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    return tmp_0

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)