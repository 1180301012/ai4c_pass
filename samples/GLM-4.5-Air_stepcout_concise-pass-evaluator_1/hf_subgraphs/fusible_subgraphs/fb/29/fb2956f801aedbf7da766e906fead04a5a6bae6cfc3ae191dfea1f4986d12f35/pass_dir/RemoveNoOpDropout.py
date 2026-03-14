import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern to match dropout with p=0.0 (no-op)
    """
    return torch.nn.functional.dropout(x, 0.0, False, False)

def replacement_args(x):
    """
    Extract arguments for replacement - just the input tensor
    """
    return (x,)

def replacement_func():
    """
    Return function that eliminates dropout by returning input directly
    """
    def eliminate_dropout(x):
        return x
    
    return eliminate_dropout