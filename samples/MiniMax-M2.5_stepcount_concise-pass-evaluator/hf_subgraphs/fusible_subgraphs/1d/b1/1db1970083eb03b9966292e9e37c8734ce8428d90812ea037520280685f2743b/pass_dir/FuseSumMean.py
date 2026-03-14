import torch


def pattern(in_0):
    """Match the computation pattern: sum(1) followed by mean((2,3), keepdim=True)"""
    tmp_0 = in_0.sum(1)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return tmp_1


def replacement_args(in_0):
    """Extract arguments needed for the replacement function"""
    return (in_0,)


@torch.fx.wrap
def optimized_sum_mean(in_0):
    """
    Optimized implementation that avoids creating intermediate tensor.
    
    Original: 
        tmp_0 = in_0.sum(1)     # [1, 2, C, H, W] -> [1, C, H, W]
        tmp_1 = tmp_0.mean((2, 3), keepdim=True)  # -> [1, C, 1, 1]
    
    Optimized:
        Compute mean over dims 1, 3, 4 to get [1, 1, C, 1, 1]
        Then squeeze the batch dimension to get [1, C, 1, 1]
    
    This avoids creating the intermediate [1, C, H, W] tensor.
    """
    # Compute mean directly over dims 1, 3, 4
    # This gives us [1, 1, C, 1, 1]
    tmp = in_0.mean(dim=(1, 3, 4), keepdim=True)
    
    # Squeeze dimension 0 (the first dimension which is 1) to get [1, C, 1, 1]
    result = tmp.squeeze(0)
    
    return result


def replacement_func():
    """Return the replacement function"""
    return optimized_sum_mean