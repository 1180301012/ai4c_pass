import torch


def pattern(in_4, in_5, in_0, in_1, in_2, in_3):
    """
    Pattern matching the full computation chain:
    1. Add: tmp_4 = in_5 + in_4
    2. Mean: tmp_5 = tmp_4.mean((2, 3))
    3. Dropout (p=0.0 no-op): tmp_6 = dropout(tmp_5, 0.0)
    4. Dropout (p=0.0 no-op): tmp_7 = dropout(tmp_6, 0.0)
    5. BatchNorm: tmp_8 = batch_norm(tmp_7, ...)
    
    We optimize by:
    - Using fused add+mean: (in_5 + in_4).mean() = in_5.mean() + in_4.mean()
    - Skipping dropout entirely (p=0.0 is no-op)
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
    def optimized_full(in_0, in_1, in_2, in_3, in_4, in_5):
        """
        Optimized version:
        - Fused add+mean: in_4.mean() + in_5.mean()
        - Skip dropout (p=0.0 is no-op)
        - Apply batch_norm
        """
        # Compute means separately then add (fused add+mean)
        mean_4 = in_4.mean(dim=(2, 3), keepdim=False)
        mean_5 = in_5.mean(dim=(2, 3), keepdim=False)
        tmp_5 = mean_4 + mean_5
        
        # Skip dropout (no-op with p=0.0)
        tmp_7 = tmp_5
        
        # Batch norm - need to call it to match return values
        tmp_8 = torch.nn.functional.batch_norm(tmp_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
        
        return tmp_8, tmp_7
    
    return optimized_full