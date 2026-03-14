import torch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Pattern matching the entire computation:
    - Add two tensors
    - Mean over spatial dims
    - Two dropout ops with p=0.0 (no-ops)
    - BatchNorm
    
    Returns both tmp_8 (batch_norm output) and tmp_7 (dropout output)
    """
    tmp_4 = in_5 + in_4
    tmp_5 = tmp_4.mean((2, 3), keepdim=False)
    tmp_6 = torch.nn.functional.dropout(tmp_5, 0.0, False, False)
    tmp_7 = torch.nn.functional.dropout(tmp_6, 0.0, False, False)
    tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return tmp_8, tmp_7


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    def optimized_full_graph(in_0, in_1, in_2, in_3, in_4, in_5):
        """
        Optimized version that:
        1. Fuses add + mean: (in_5 + in_4).mean() = in_5.mean() + in_4.mean()
        2. Skips dropout (no-op with p=0.0)
        3. Applies batch_norm
        
        The key optimization is computing the mean of each tensor separately
        and adding, which can be better parallelized.
        """
        # Compute means separately - enables better memory access patterns
        mean_4 = in_4.mean(dim=(2, 3), keepdim=False)
        mean_5 = in_5.mean(dim=(2, 3), keepdim=False)
        
        # Combine means (this is mathematically equivalent to (in_4 + in_5).mean())
        tmp_5 = mean_4 + mean_5
        
        # Dropout with p=0.0 is a no-op, skip it
        tmp_7 = tmp_5  # tmp_6 and tmp_7 are no-ops
        
        # Apply batch norm
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        
        return tmp_8, tmp_7
    
    return optimized_full_graph