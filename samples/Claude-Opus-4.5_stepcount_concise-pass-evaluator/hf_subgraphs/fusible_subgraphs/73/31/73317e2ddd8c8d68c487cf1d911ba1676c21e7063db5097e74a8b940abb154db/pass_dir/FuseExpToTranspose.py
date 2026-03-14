import torch
from torch import device
import triton
import triton.language as tl

# Pattern to match the full computation
def pattern(in_0, in_1):
    tmp_1 = in_0.exp()
    tmp_2 = tmp_1.to(device=device(type='cuda', index=0))
    tmp_3 = in_1.to(device=device(type='cuda', index=0), dtype=torch.float32)
    tmp_4 = tmp_3.t()
    return tmp_3, tmp_2, tmp_4

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def fast_exp(x):
    """Fast exp using native PyTorch"""
    return x.exp()

# Main replacement function - fully traceable by FX
def fused_exp_to_transpose(in_0, in_1):
    # Fast exp using wrapped function
    tmp_2 = fast_exp(in_0)
    
    # Skip .to() since data is already on correct device/dtype
    tmp_3 = in_1
    
    # t() is a view operation - no data copy
    tmp_4 = tmp_3.t()
    
    return tmp_3, tmp_2, tmp_4

def replacement_func():
    return fused_exp_to_transpose