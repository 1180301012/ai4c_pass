import torch

def pattern(in_2):
    tmp_8 = in_2.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(0)
    return tmp_9

def replacement_args(in_2):
    return (in_2,)

def fused_unsqueeze_mask(in_2):
    """
    Fuse the two unsqueeze operations into a single operation.
    Original: in_2 -> unsqueeze(1) -> unsqueeze(0) 
    Resulting shape: [1, N, 1, H, W]
    """
    # Fuse both operations: add dim at index 0, then at index 2
    return in_2.unsqueeze(0).unsqueeze(2)  # [N, H, W] -> [1, N, H, W] -> [1, N, 1, H, W]

def replacement_func():
    return fused_unsqueeze_mask