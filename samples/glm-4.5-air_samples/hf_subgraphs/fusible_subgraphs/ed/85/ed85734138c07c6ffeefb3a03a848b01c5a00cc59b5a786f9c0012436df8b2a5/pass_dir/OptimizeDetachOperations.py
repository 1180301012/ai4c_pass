import torch

# Pattern matching function - matches the three detach operations (accepts all inputs)
def pattern(in_0, in_1, in_2, tmp_0):
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach()
    tmp_3 = tmp_0.detach()
    return (tmp_1, tmp_2, tmp_3)

# Argument extraction function
def replacement_args(in_0, in_1, in_2, tmp_0):
    return (in_1, in_2, tmp_0)

# Optimized detach operations
@torch.fx.wrap
def optimized_detach(in_1, in_2, tmp_0):
    # All detach operations can be done in parallel
    tmp_1 = in_1.detach()
    tmp_2 = in_2.detach() 
    tmp_3 = tmp_0.detach()
    
    return (tmp_1, tmp_2, tmp_3)

# Replacement function
def replacement_func():
    return optimized_detach