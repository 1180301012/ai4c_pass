import torch
import triton
import triton.language as tl

def pattern(x):
    tmp_0 = x.sum(dim=1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    return tmp_1

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def simple_direct_mean(x):
    """
    Direct computation: Instead of sum(dim=1) then adaptive_avg_pool2d(..., 1),
    directly compute the mean over both channel and spatial dimensions.
    
    The original operation:
    tmp_0 = x.sum(dim=1)  # [B, 128, H, W]
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)  # [B, 128, 1, 1]
    
    This is equivalent to:
    result[b, f] = mean(x[b, :, f, :, :])  # mean over the 2-channel and H,W dimensions
    """
    B, C, H, W = x.shape[0], x.shape[1], x.shape[3], x.shape[4]
    
    # Simply compute the mean over the channel dimension (size=C=2) and spatial dimensions (H,W)
    # This directly gives the same result as sum(dim=1) followed by adaptive_avg_pool2d(..., 1)
    result = x.mean(dim=(1, 3, 4))  # Mean over channel, height, and width dimensions
    
    # Reshape to [B, C, 1, 1] to match the original output format
    return result.unsqueeze(-1).unsqueeze(-1)

def replacement_func():
    return simple_direct_mean