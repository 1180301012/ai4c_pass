import torch

@torch.fx.wrap
def optimized_zeros(tmp_1, shape_0, shape_1):
    # Direct zeros creation - torch.zeros is already optimized
    # But we ensure we use the correct dtype and device
    return torch.zeros((shape_0, shape_1), dtype=tmp_1.dtype, device=tmp_1.device)

def pattern(tmp_1, shape_0, shape_1):
    tmp_4 = tmp_1.new_zeros((shape_0, shape_1))
    return tmp_4

def replacement_args(tmp_1, shape_0, shape_1):
    return (tmp_1, shape_0, shape_1)

def replacement_func():
    return optimized_zeros