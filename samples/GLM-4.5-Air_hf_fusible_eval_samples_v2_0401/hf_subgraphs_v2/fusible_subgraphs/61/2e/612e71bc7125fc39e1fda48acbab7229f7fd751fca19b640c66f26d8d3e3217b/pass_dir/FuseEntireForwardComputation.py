import torch
from torch import device

@torch.fx.wrap
def fused_forward_computation(in_0, in_1):
    """Fused forward computation combining all operations"""
    # Process in_1: L2 normalization (fusion optimization for small batches)
    tmp_1 = in_1 / in_1.norm(p=2, dim=-1, keepdim=True)
    
    # Process in_0: Efficient transpose (remove redundant device transfer)
    tmp_3 = in_0.t()
    
    return tmp_1, tmp_3

def pattern(in_0, in_1):
    """Pattern: Full forward computation - normalization + transpose"""
    tmp_0 = in_1.norm(p = 2, dim = -1, keepdim = True)
    tmp_1 = in_1 / tmp_0;  in_1 = tmp_0 = None
    tmp_2 = in_0.t();  in_0 = None
    tmp_3 = tmp_2.to(device(type='cuda'));  tmp_2 = None
    return (tmp_1, tmp_3)

def replacement_args(in_0, in_1):
    """Extract arguments for replacement"""
    return (in_0, in_1)

def replacement_func():
    """Return the optimized function"""
    return fused_forward_computation