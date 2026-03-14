import torch
import triton
import triton.language as tl

def pattern(x):
    """
    Pattern: flatten(2) followed by transpose(1, 2)
    """
    flattened = x.flatten(2)
    transposed = flattened.transpose(1, 2)
    return transposed

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_flatten_transpose(x):
    """
    Optimized flatten + transpose using PyTorch's native operations
    Directly use permute + reshape which is highly optimized
    """
    batch, C, H, W = x.shape
    # Permute to [B, H, W, C] then reshape to [B, H*W, C]
    # This is more efficient than flatten + transpose
    return x.permute(0, 2, 3, 1).contiguous().view(batch, H * W, C)

def replacement_func():
    return fused_flatten_transpose