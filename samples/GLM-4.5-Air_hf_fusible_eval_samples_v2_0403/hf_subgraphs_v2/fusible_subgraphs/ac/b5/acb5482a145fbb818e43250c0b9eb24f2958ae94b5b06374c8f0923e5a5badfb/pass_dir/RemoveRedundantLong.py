import torch
import triton
import triton.language as tl

def pattern(attention_mask):
    """Match the redundant .long() operation on int64 tensors"""
    converted = attention_mask.long()
    return converted

def replacement_args(attention_mask):
    """Extract the attention_mask tensor"""
    return (attention_mask,)

@torch.fx.wrap
def kernel_wrapper(attention_mask):
    """Simply return the original tensor since the conversion is redundant"""
    return attention_mask

def replacement_func():
    """Return the kernel wrapper function"""
    return kernel_wrapper