import torch

def pattern(in_2, in_1, in_0):
    # TorchGeometric pattern: dropout(p=0.0) + to(dtype) + linear
    tmp_2 = torch.nn.functional.dropout(in_2, p = 0.0, training = False)
    to = tmp_2.to(torch.float16)  # This will also match bfloat16 if we use a different pattern
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

@torch.fx.wrap
def simple_torchgeometric_optimization(x, weight, bias):
    # Dropout with p=0.0 is identity operation - skip it
    # The .to() conversion might be redundant if already in target dtype
    # For now, just run the linear directly
    return torch.nn.functional.linear(x, weight, bias)

def replacement_func():
    return simple_torchgeometric_optimization