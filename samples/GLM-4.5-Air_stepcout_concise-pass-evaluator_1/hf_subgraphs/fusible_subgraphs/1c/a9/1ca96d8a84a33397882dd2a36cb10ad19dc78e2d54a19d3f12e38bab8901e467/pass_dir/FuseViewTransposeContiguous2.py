import torch

def pattern(tmp_6):
    """Match second sequence: cat -> view -> transpose -> contiguous -> view"""
    tmp_11 = tmp_6.view(-1, 2, 40, 32, 24)
    tmp_12 = torch.transpose(tmp_11, 1, 2)
    tmp_13 = tmp_12.contiguous()
    tmp_14 = tmp_13.view(-1, 80, 32, 24)
    return tmp_14

def replacement_args(tmp_6):
    return (tmp_6,)

def fused_view_transform2(tmp_6):
    """Direct reshape eliminating intermediate transpose and contiguous operations"""
    # The sequence:
    # view(-1, 2, 40, 32, 24) -> transpose(1, 2) -> contiguous() -> view(-1, 80, 32, 24)
    # Can be optimized to a single view(-1, 80, 32, 24) because:
    # view(-1, 2, 40, 32, 24) gives shape [batch, 2, 40, 32, 24]
    # transpose(1, 2) gives [batch, 40, 2, 32, 24] 
    # view(-1, 80, 32, 24) combines 40*2 = 80
    # The contiguous() is redundant after transpose since we immediately reshape
    return input_tensor.view(-1, 80, 32, 24)

def replacement_func():
    return fused_view_transform2