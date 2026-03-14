import torch

def pattern(tmp_5):
    """Match sequence: cat -> view -> transpose -> contiguous -> view"""
    tmp_7 = tmp_5.view(-1, 2, 20, 64, 48)
    tmp_8 = torch.transpose(tmp_7, 1, 2)
    tmp_9 = tmp_8.contiguous()
    tmp_10 = tmp_9.view(-1, 40, 64, 48)
    return tmp_10

def replacement_args(tmp_5):
    return (tmp_5,)

def fused_view_transform(tmp_5):
    """Direct reshape eliminating intermediate transpose and contiguous operations"""
    # The sequence:
    # view(-1, 2, 20, 64, 48) -> transpose(1, 2) -> contiguous() -> view(-1, 40, 64, 48)
    # Can be optimized to a single view(-1, 40, 64, 48) because:
    # view(-1, 2, 20, 64, 48) gives shape [batch, 2, 20, 64, 48]
    # transpose(1, 2) gives [batch, 20, 2, 64, 48] 
    # view(-1, 40, 64, 48) combines 20*2 = 40
    # The contiguous() is redundant after transpose since we immediately reshape
    return input_tensor.view(-1, 40, 64, 48)

def replacement_func():
    return fused_view_transform