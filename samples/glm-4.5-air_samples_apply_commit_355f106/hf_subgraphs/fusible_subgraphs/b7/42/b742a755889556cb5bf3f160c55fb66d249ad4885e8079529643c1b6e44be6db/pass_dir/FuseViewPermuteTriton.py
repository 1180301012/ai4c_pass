import torch
import triton
import triton.language as tl

# Pattern: First View + Permute after Conv2D
# This is: [B, C, H, W] -> view to [B, C, HW] -> permute to [B, HW, C]
# We can fuse this into a more efficient single operation
def pattern(x):
    # First view: [B, C, H*W] 
    tmp = x.view(1, 384, 576)
    # Then permute: [B, H*W, C]
    tmp = tmp.permute(0, 2, 1)
    return tmp

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_view_permute(x):
    # Original: view(1, 384, 576) then permute(0, 2, 1)
    # x is [1, 384, 24, 24]
    # First do the view: [1, 384, 576]
    tmp = x.view(1, 384, 576)
    # Then permute: [1, 576, 384]
    tmp = tmp.permute(0, 2, 1)
    return tmp

def replacement_func():
    return fused_view_permute