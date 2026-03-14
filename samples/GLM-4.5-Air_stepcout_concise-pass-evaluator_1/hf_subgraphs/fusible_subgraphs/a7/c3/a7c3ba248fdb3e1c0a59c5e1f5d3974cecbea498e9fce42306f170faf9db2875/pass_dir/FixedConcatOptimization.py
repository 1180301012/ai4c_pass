import torch

@torch.fx.wrap
def create_simple_fixed_tensor():
    # Simple PyTorch tensor creation
    return torch.tensor([-1], dtype=torch.int64, device='cuda')

# Pattern matching function - match tensor creation
def pattern():
    tmp_4 = torch.as_tensor([-1], dtype=torch.int64)
    return tmp_4

# Argument extraction function (no args needed for this pattern)
def replacement_args():
    return ()

# Replacement function (returns function reference)
def replacement_func():
    return create_simple_fixed_tensor