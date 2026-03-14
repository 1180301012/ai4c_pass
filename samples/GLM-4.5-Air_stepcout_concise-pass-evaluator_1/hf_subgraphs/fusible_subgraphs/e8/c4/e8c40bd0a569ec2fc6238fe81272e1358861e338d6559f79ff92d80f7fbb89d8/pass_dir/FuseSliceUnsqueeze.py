import torch


# Pattern matching function - matches the slice + unsqueeze pattern
# This pattern: select channel 0, then unsqueeze at dimension 1
def pattern(in_0):
    tmp_0 = in_0
    tmp_1 = tmp_0[slice(None, None, None), 0]
    tmp_2 = torch.unsqueeze(tmp_1, 1)
    return tmp_2


# Argument extraction function
def replacement_args(in_0):
    return (in_0,)


def fused_slice_unsqueeze(x):
    """Optimized slice + unsqueeze fusion using PyTorch.
    
    Replaces: x[:, 0] followed by unsqueeze(dim=1)
    With: x[:, 0:1, :, :] which is more efficient - avoids intermediate tensor allocation
    """
    # Use expanded slice to directly get [B, 1, H, W] instead of [B, H, W] -> [B, 1, H, W]
    # This avoids creating an intermediate tensor
    return x[:, 0:1, :, :]


def replacement_func():
    return fused_slice_unsqueeze