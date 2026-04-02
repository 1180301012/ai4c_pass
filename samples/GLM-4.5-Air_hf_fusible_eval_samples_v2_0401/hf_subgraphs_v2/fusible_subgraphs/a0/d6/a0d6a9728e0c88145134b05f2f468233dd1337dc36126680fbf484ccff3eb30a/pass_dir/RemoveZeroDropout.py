import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern to match: dropout with p=0.0 and training=False"""
    return torch.nn.functional.dropout(input_tensor, p=0.0, training=False)

def replacement_args(input_tensor):
    """Extract input tensor for the no-op replacement"""
    return (input_tensor,)

def replacement_func():
    """Return a no-op function that just passes the input through"""
    def noop_dropout(input_tensor):
        return input_tensor
    return noop_dropout