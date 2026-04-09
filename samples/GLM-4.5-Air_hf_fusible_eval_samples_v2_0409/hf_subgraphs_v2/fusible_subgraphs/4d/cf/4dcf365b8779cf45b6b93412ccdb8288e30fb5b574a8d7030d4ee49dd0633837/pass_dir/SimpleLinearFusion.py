import torch

def pattern(in_2, in_1, in_0):
    # BigBird pattern: dropout(0.1) + linear
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return linear

def replacement_args(in_2, in_1, in_0):
    return (in_2, in_1, in_0)

# Just use regular torch.nn.functional.linear as a simple replacement
# This focuses on correctness first, then we can optimize with Triton later
@torch.fx.wrap
def simple_linear_fusion(x, weight, bias):
    # Apply dropout with scale
    dropout_scale = 1.0 / (1.0 - 0.1)  # Scale for training
    x_dropped = x * dropout_scale
    mask = torch.rand_like(x) > 0.1  # Dropout mask
    x_dropped = x_dropped * mask
    
    # Apply linear transformation
    return torch.nn.functional.linear(x_dropped, weight, bias)

def replacement_func():
    return simple_linear_fusion