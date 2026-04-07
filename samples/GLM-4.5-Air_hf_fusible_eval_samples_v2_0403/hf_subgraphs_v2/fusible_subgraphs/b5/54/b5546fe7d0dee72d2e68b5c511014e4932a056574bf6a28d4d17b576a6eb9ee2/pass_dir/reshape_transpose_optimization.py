import torch

def pattern(tmp_3):
    tmp_4 = tmp_3.reshape(1, 8, 19, 196)
    tmp_5 = tmp_4.transpose(-1, -2)
    return tmp_5

def replacement_args(tmp_3):
    return (tmp_3,)

@torch.fx.wrap
def optimized_reshape_transpose(tmp_3):
    # Use view instead of reshape for better performance
    tmp_4 = tmp_3.view(1, 8, 19, 196)
    
    # Use permute for explicit transpose which can be more efficient
    tmp_5 = tmp_4.permute(0, 3, 1, 2)
    
    return tmp_5

def replacement_func():
    return optimized_reshape_transpose