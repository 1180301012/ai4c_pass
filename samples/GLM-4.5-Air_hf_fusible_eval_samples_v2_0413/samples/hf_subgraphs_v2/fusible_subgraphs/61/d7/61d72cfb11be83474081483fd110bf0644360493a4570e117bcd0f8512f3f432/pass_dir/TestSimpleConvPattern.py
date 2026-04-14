import torch

# Pattern matching function 
def pattern(in_1, in_0):
    """
    Simple test pattern - just match conv2d
    """
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return conv2d

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Replacement function
def replacement_func():
    # Simple replacement - just return the same computation
    def simple_replacement(in_1, in_0):
        return torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    return simple_replacement