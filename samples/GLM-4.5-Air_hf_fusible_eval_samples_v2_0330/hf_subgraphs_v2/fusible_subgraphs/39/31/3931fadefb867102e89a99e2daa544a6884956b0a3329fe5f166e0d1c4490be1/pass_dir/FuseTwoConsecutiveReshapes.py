import torch
import triton
import triton.language as tl

def pattern(x):
    """Match two consecutive reshape operations that can be fused"""
    tmp_1 = x.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    return tmp_2

def replacement_args(x):
    """Extract input tensor for the fused reshape operation"""
    return (x,)

@torch.fx.wrap  
def fused_reshape(x):
    """Direct reshape from [1, 124, 1536] to [1, 248, 768]"""
    # This is just a metadata-only operation, no data movement needed
    return x.reshape(1, 248, 768)

def replacement_func():
    return fused_reshape