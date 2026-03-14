import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: flatten(2) followed by transpose(1, 2)
    Input x shape: [batch, channels, h, w]
    After flatten(2): [batch, channels, h*w]
    After transpose(1, 2): [batch, h*w, channels]
    """
    tmp = x.flatten(2)
    result = tmp.transpose(1, 2)
    return result

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_flatten_transpose(x):
    """
    Optimized implementation using PyTorch's efficient operations
    Input: [batch, channels, h, w]
    Output: [batch, h*w, channels]
    """
    # Use contiguous permute for better performance
    # [batch, channels, h, w] -> [batch, h, w, channels] -> [batch, h*w, channels]
    return x.permute(0, 2, 3, 1).flatten(1, 2)

def replacement_func():
    return fused_flatten_transpose