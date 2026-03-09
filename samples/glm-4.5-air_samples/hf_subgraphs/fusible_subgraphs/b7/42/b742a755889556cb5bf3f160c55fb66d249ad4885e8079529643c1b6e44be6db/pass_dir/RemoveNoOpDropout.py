import torch
import triton
import triton.language as tl

def pattern(x):
    # Match torch.nn.functional.dropout with p=0.0
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    return (x,)

# No-op - just return the input directly
@torch.fx.wrap
def no_op_dropout(x):
    return x

def replacement_func():
    return no_op_dropout