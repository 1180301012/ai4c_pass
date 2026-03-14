import torch

# Define sym_sum and add it to torch namespace
def sym_sum(values):
    """Symbolic sum - computes sum of list elements"""
    result = 0
    for v in values:
        result = result + v
    return result

# Add to torch namespace if not present
if not hasattr(torch, 'sym_sum'):
    torch.sym_sum = sym_sum