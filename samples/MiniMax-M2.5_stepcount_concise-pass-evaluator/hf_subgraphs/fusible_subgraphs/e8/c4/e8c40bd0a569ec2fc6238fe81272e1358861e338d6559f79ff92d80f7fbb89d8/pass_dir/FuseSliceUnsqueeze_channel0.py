import torch

# Pattern matching function - matches slice at channel 0 followed by unsqueeze at dim 1
def pattern(in_0):
    """
    Match the computation pattern: slice channel 0 then unsqueeze at dimension 1
    This extracts the first channel and adds a channel dimension back.
    Input shape: [B, C, H, W]
    Output shape: [B, 1, H, W]
    """
    tmp_0 = in_0
    tmp_1 = tmp_0[slice(None, None, None), 0]
    tmp_2 = torch.unsqueeze(tmp_1, 1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def optimized_slice_unsqueeze(in_0):
    """
    Optimized implementation using reshape/view.
    
    Original pattern:
        tmp_1 = tmp_0[slice(None, None, None), 0]  # Shape: [B, H, W]
        tmp_2 = torch.unsqueeze(tmp_1, 1)          # Shape: [B, 1, H, W]
    
    This is equivalent to:
        output = in_0[:, 0:1, :, :]  # Shape: [B, 1, H, W]
    
    We use direct slicing which can be better optimized by PyTorch's
    fused kernel implementations.
    """
    # Using contiguous slice with range - allows PyTorch to use optimized kernels
    # The key optimization is using 0:1 instead of separate index + unsqueeze
    return in_0[:, 0:1, :, :].contiguous()


def replacement_func():
    return optimized_slice_unsqueeze