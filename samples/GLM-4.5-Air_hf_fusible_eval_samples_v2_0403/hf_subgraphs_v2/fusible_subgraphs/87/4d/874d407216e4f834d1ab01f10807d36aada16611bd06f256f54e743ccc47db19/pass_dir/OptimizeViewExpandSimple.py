import torch

@torch.fx.wrap
def optimized_expand(in_0, tmp_1):
    # Direct broadcast without intermediate view and expand
    # This is more efficient as it avoids creating temporary tensors
    if in_0.dim() == 1:
        # Directly broadcast to match tmp_1 shape
        return in_0.unsqueeze(1).expand_as(tmp_1)
    else:
        return in_0.expand_as(tmp_1)

def pattern(in_0, tmp_1):
    tmp_2 = in_0.view((-1, 1))
    tmp_3 = tmp_2.expand_as(tmp_1)
    return tmp_3

def replacement_args(in_0, tmp_1):
    return (in_0, tmp_1)

def replacement_func():
    return optimized_expand