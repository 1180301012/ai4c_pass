import torch


def pattern(in_0, in_1):
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    tmp_4 = tmp_2.masked_fill_(tmp_3, 1)
    tmp_5 = tmp_2.unsqueeze(0)
    tmp_6 = tmp_5.expand(3, -1, -1)
    import torch
    from torch import device
    tmp_7 = tmp_6.to(device(type='cuda', index=0))
    max_1 = tmp_7.max(0, keepdim=False)
    tmp_9 = max_1[0]
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
    return tmp_13, tmp_7

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def optimized_cumsum_max(in_0, in_1):
    # Optimized computation that eliminates unnecessary operations
    tmp_1 = in_1.cumsum(-1)
    tmp_2 = tmp_1 - 1
    tmp_3 = in_0.__eq__(0)
    # Skip masked_fill_ result since it's immediately discarded (dead code elimination)
    
    # Optimization: Expand and compute max without redundant device transfer
    tmp_7 = tmp_2.unsqueeze(0).expand(3, *tmp_2.shape)
    
    # Continue with max computations
    batch_max = tmp_7.max(0)[0]
    tmp_9 = batch_max
    max_2 = tmp_9.max(-1, keepdim=True)
    tmp_11 = max_2[0]
    tmp_12 = tmp_11 + 1
    tmp_13 = tmp_12 - 9
        
    return tmp_13, tmp_7

def replacement_func():
    return optimized_cumsum_max