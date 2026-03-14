import torch

def pattern(x, weight, bias, in_3):
    """Fused linear and mean operations pattern matching"""
    # The original computation has:
    # tmp_2 = torch.nn.functional.linear(x, weight, bias)
    # tmp_3 = in_3.mean(-2)
    # return (tmp_2, tmp_3)
    
    # Match both operations together
    linear_out = torch.nn.functional.linear(x, weight, bias)
    mean_out = in_3.mean(-2)
    return linear_out, mean_out

def replacement_args(x, weight, bias, in_3):
    """Extract arguments for the fused operations"""
    return (x, weight, bias, in_3)

def fused_operations(x, weight, bias, in_3):
    """
    Simple fused implementation that matches the original computation exactly.
    The goal is to match the pattern and provide a working solution.
    """
    # Linear operation - use direct equivalent of torch.nn.functional.linear()
    linear_out = x @ weight.transpose(-1, -2) + bias
    
    # Mean operation - equivalent to in_3.mean(-2)
    mean_out = in_3.mean(dim=-2)
    
    return linear_out, mean_out

def replacement_func():
    """Return the fused computation function"""
    return fused_operations