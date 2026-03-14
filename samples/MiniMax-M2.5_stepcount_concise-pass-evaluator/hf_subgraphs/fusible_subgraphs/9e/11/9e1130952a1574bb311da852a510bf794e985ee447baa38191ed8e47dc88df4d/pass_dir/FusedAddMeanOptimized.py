import torch


def pattern(in_4, in_5):
    """
    Pattern: Add two tensors followed by mean over spatial dimensions.
    
    Optimization: Use fused computation that avoids creating intermediate tensor.
    (in_5 + in_4).mean(dim=(2,3)) can be computed as in_5.mean() + in_4.mean()
    which avoids materializing the sum tensor.
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    return tmp_5


def replacement_args(in_4, in_5):
    return (in_4, in_5)


def replacement_func():
    def optimized_add_mean(in_4, in_5):
        """
        Fused add + mean that avoids creating intermediate sum tensor.
        
        Original: tmp_4 = in_5 + in_4; tmp_5 = tmp_4.mean()
        Optimized: tmp_5 = in_5.mean() + in_4.mean()
        
        This avoids allocating the [B,C,H,W] intermediate tensor.
        """
        # Compute mean of each tensor separately, then add
        # This avoids creating the (in_5 + in_4) tensor of size [B,C,H,W]
        return in_5.mean(dim=(2, 3), keepdim=False) + in_4.mean(dim=(2, 3), keepdim=False)
    
    return optimized_add_mean