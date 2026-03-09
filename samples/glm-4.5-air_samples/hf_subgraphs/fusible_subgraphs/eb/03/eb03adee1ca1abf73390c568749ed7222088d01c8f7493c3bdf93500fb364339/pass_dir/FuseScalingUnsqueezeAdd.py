import torch

# Pattern matching function - matches scaling + unsqueeze + addition
def pattern(in_0, in_1):
    tmp_0 = in_1 * 0.1767766952966369
    tmp_1 = in_0.unsqueeze(2)
    tmp_2 = tmp_0 + tmp_1
    return tmp_2

# Argument extraction function
def replacement_args(in_0, in_1):
    return (in_0, in_1)

# More optimized fusion - minimize memory operations and let PyTorch optimize
@torch.fx.wrap  
def fused_scaling_unsqueeze_add(in_0, in_1, scale_factor=0.1767766952966369):
    # Single fused operation: scale, broadcast, and add in one step
    # This allows PyTorch to optimize memory access patterns
    return in_1 * scale_factor + in_0.unsqueeze(2)

# Replacement function
def replacement_func():
    return fused_scaling_unsqueeze_add