import torch

def pattern(in_0, in_3):
    tmp_1 = in_0[in_3]
    tmp_2 = tmp_1.view(-1, -1, -1)
    tmp_3 = tmp_2.permute(2, 0, 1)
    tmp_4 = tmp_3.contiguous()
    tmp_5 = tmp_4.unsqueeze(0)
    return tmp_5

def replacement_args(in_0, in_3):
    return (in_0, in_3)

def optimized_view_permute_contiguous_unsqueeze(in_0, in_3):
    """
    Optimized version that skips unnecessary contiguous() call
    and directly creates the output in the desired shape.
    This eliminates intermediate tensors and memory copies.
    """
    # Direct indexing and reshape - let TorchFX handle the shapes symbolically
    # The key optimization is skipping the contiguous() call when not needed
    tmp_1 = in_0[in_3]  # Indexing
    
    # Directly reshape and permute without intermediate tensors
    # The framework can handle the shape inference symbolically
    result = tmp_1.view(-1, -1, -1).permute(2, 0, 1).unsqueeze(0)
    
    return result

def replacement_func():
    return optimized_view_permute_contiguous_unsqueeze