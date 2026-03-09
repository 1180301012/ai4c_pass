import torch
import triton
import triton.language as tl


def pattern(in_0):
    """Match the pattern: sum(dim=1) followed by adaptive_avg_pool2d(1)"""
    tmp_0 = in_0.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1


def replacement_args(in_0):
    """Extract the input tensor"""
    return (in_0,)


@torch.fx.wrap
def fused_sum_adaptive_avg_pool2d_wrapper(in_0):
    """Optimized wrapper using PyTorch's mean operation.
    
    The original pattern:
        tmp_0 = in_0.sum(dim=1)   # [B, C, D, H, W] -> [B, D, H, W]
        tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)  # [B, D, H, W] -> [B, D, 1, 1]
    
    Is mathematically equivalent to:
        mean over dim 1, 3, 4 = (1/(C*H*W)) * sum over C, H, W
        
    Compute more efficiently with a single mean, avoiding intermediate tensors.
    """
    B, C, D, H, W = in_0.shape
    
    # Reshape input to merge the dimensions we want to reduce over
    # Input: [B, C, D, H, W] -> [B, D, C*H*W]
    in_reshaped = in_0.reshape(B, C, D, H * W)
    in_reshaped = in_reshaped.transpose(1, 2)  # [B, D, C, H*W]
    in_reshaped = in_reshaped.reshape(B, D, C * H * W)
    
    # Compute mean over the last dimension
    result = in_reshaped.mean(dim=2, keepdim=True)  # [B, D, 1]
    
    # Reshape to [B, D, 1, 1]
    result = result.reshape(B, D, 1, 1)
    
    return result


def replacement_func():
    """Return the replacement function."""
    return fused_sum_adaptive_avg_pool2d_wrapper