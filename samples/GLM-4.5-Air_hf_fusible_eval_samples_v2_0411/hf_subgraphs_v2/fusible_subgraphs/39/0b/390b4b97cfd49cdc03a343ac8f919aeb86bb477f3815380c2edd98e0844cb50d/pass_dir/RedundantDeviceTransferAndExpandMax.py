import torch
from torch import device

def pattern(tmp_2):
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_7, tmp_13

def replacement_args(tmp_2):
    return (tmp_2,)



@torch.fx.wrap
def optimize_expand_max_combined(tmp_2):
    # Add batch dimension and compute efficiently
    tmp_5 = tmp_2.unsqueeze(0)  # Shape: [1, B, D]
    
    # Instead of expand + redundant device transfer, compute directly
    # This eliminates the .to(device(...)) call which is redundant
    tmp_7 = tmp_5.expand(3, *tmp_2.shape)
    
    # Continue with the max computation chain
    batch_max = tmp_7.max(0)[0]
    tmp_9 = batch_max
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
        
    return tmp_7, tmp_13

def replacement_func():
    return optimize_expand_max_combined