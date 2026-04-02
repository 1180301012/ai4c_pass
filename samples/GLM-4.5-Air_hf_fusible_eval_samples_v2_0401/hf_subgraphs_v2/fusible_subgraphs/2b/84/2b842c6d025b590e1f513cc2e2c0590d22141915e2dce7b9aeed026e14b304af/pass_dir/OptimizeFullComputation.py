import torch

def pattern(in_0, in_1):
    # This matches the entire computation from inputs to outputs
    tmp_1 = in_0 * 1000000.0
    tmp_2 = in_1 - tmp_1
    split = tmp_2.split(1, dim = -1)
    tmp_4 = split[0]
    tmp_5 = split[1]
    tmp_6 = tmp_4.squeeze(-1)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_5.squeeze(-1)
    tmp_9 = tmp_8.contiguous()
    return (tmp_7, tmp_9)

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@torch.fx.wrap
def optimized_forward(in_0, in_1):
    # Optimized version that eliminates unnecessary intermediate operations
    # Move in_0 to GPU once and convert to proper dtype
    in_0_gpu = in_0.to(in_1.device).to(in_1.dtype)
    
    # Direct computation without intermediate tensors
    # 1. Broadcast and arithmetic in one step
    result = in_1 - in_0_gpu * 1000000.0
    
    # 2. Extract columns directly using indexing (fuses split + squeeze)
    # Instead of split(1, dim=-1) followed by squeeze(-1), use direct indexing
    col0 = result[..., 0].contiguous()
    col1 = result[..., 1].contiguous()
    
    return col0, col1

def replacement_func():
    return optimized_forward