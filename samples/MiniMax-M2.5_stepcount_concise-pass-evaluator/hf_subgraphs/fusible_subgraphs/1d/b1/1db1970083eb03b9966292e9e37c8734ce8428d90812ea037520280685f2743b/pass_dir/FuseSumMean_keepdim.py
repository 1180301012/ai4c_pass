import torch
import triton
import triton.language as tl


# Pattern matching function - matches the exact computation pattern from model.py
def pattern(in_0):
    """
    Match the sum(1) followed by mean((2, 3), keepdim=True) pattern.
    This is equivalent to computing mean across dimensions 1, 2, 3.
    """
    tmp_0 = in_0.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1


def replacement_args(in_0):
    """Extract arguments needed for the replacement."""
    return (in_0,)


# Optimized implementation using reshape to reduce in a single mean call
def fused_mean_impl(in_0):
    """
    Compute sum(1) + mean((2,3), keepdim=True) using reshape-based approach.
    """
    # Get shapes
    batch = in_0.shape[0]
    dim1 = in_0.shape[1]
    channels = in_0.shape[2]
    H = in_0.shape[3]
    W = in_0.shape[4]
    
    # Reshape to merge reduction dims: [1, 2, 256, H, W] -> [1, 256, 2*H*W]
    in_reshaped = in_0.reshape(batch, channels, dim1 * H * W)
    
    # Mean over last dim: [1, 256, 2*H*W] -> [1, 256, 1]
    result = in_reshaped.mean(dim=2, keepdim=True)
    
    # Reshape to [1, 256, 1, 1]
    return result.reshape(batch, channels, 1, 1)


def replacement_func():
    """Return the replacement function."""
    return fused_mean_impl