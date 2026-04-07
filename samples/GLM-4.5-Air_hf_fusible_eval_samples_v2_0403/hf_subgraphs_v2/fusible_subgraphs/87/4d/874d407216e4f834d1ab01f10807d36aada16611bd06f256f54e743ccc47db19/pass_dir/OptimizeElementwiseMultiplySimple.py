import torch

@torch.fx.wrap
def optimized_multiply(in_1, in_2):
    # Direct multiplication with broadcasting
    # PyTorch handles broadcasting efficiently, so we just need
    # to ensure the shapes are compatible
    if in_1.dim() == 1:
        # Automatically broadcasts correctly
        return in_1.unsqueeze(1) * in_2
    else:
        return in_1 * in_2

def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1

def replacement_args(in_1, in_2):
    return (in_1, in_2)

def replacement_func():
    return optimized_multiply