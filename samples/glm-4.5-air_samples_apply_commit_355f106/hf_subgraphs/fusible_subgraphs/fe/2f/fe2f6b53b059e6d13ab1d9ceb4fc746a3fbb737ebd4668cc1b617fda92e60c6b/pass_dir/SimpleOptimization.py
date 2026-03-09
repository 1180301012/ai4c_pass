import torch

def pattern(x):
    return torch.nn.functional.relu(x, inplace=True)

def replacement_args(x):
    return (x,)

def simple_relu(x):
    return torch.nn.functional.relu(x, inplace=True)

@torch.fx.wrap
def simple_relu_wrapper(x):
    return simple_relu(x)

def replacement_func():
    return simple_relu_wrapper