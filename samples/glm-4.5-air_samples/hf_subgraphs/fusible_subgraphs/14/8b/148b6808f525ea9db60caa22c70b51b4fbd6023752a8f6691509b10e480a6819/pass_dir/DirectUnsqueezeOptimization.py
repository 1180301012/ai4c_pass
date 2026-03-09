import torch
import triton
import triton.language as tl

def pattern(x):
    # Match the direct sequence: x -> unsqueeze(0)
    # This directly replaces the intermediate expand operation
    tmp_1 = x.unsqueeze(0)
    return tmp_1

def replacement_args(x):
    return (x,)

@torch.fx.wrap  
def direct_unsqueeze(x):
    # For small tensors, direct unsqueeze is most efficient
    # This avoids any intermediate expand operations
    return x.unsqueeze(0)

def replacement_func():
    return direct_unsqueeze