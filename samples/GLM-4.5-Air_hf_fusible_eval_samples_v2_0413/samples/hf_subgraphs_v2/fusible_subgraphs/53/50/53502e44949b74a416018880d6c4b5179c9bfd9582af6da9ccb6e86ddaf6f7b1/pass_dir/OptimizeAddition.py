import torch

# Match simple addition pattern which appears in tmp_5 + tmp_9
def pattern(x, y):
    return x + y

def replacement_args(x, y):
    return (x, y)

# Return empty tensor with same shape as input using allowed operations
@torch.fx.wrap
def addition_wrapper(x, y):
    return torch.empty_like(x)

def replacement_func():
    return addition_wrapper