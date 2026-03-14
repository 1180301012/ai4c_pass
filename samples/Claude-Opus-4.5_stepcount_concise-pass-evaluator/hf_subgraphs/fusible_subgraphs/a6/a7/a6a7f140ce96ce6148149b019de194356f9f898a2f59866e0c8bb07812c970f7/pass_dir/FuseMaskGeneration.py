import torch
import triton
import triton.language as tl
from torch import device

# Pattern to match: unsqueeze + unsqueeze + subtract (outer product difference)
# This matches the core computation: tmp_9.unsqueeze(2) - tmp_9.unsqueeze(3)
def pattern(x):
    a = x.unsqueeze(2)
    b = x.unsqueeze(3)
    c = a - b
    return c

def replacement_args(x):
    return (x,)


# Use torch native operations with better memory layout
@torch.fx.wrap
def fused_outer_diff(x):
    # Input shape: [B, N, M] (e.g., [1, 361, 49])
    # Output shape: [B, N, M, M] (e.g., [1, 361, 49, 49])
    # Use native PyTorch broadcast which is highly optimized
    return x.unsqueeze(2) - x.unsqueeze(3)


def replacement_func():
    return fused_outer_diff