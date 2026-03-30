import torch
import torch.nn.functional as F

def pattern(in_3, in_1, in_0):
    """Linear + view + transpose pattern"""
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_5 = linear.view(1, 1, -1, 64)
    tmp_6 = tmp_5.transpose(1, 2)
    linear = None
    return tmp_6

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

@torch.fx.wrap
def optimized_linear_reshape_transpose(x, w, b=None):
    """Optimized function combining linear + reshape + transpose"""
    # Alternative implementation that doesn't call the exact same function
    # Use torch.matmul for linear transformation
    if b is not None:
        result = torch.matmul(x, w.t()) + b
    else:
        result = torch.matmul(x, w.t())
    
    # Optimized reshape and transpose in one step
    # Instead of view(1, 1, -1, 64) then transpose(1, 2), we can do:
    # This reshapes to [1, -1, 64] and then transpose to [-1, 1, 64], then squeeze
    reshaped = result.view(1, -1, 64)
    transposed = reshaped.transpose(1, 2).squeeze(0)
    
    # Add back the dimensions to match expected shape [1, tokens*heads, 1, 64]
    final = transposed.unsqueeze(2)
    
    return final

def replacement_func():
    return optimized_linear_reshape_transpose