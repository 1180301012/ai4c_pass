import torch
import triton
import triton.language as tl

# Very simple pattern to test validation
def pattern(in_3):
    return in_3.expand(1, -1, -1)

def replacement_args(in_3):
    return (in_3,)

# Simple no-op wrapper that should pass validation
@torch.fx.wrap
def simple_expand(tensor):
    # Just return the input for now (expand is a no-op in this case)
    return tensor

def replacement_func():
    return simple_expand