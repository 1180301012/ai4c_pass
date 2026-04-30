import torch


def pattern(in_0, in_1, in_2):
    """Pattern: linear(in_2, in_1, in_0)
    
    This matches any linear operation: torch.nn.functional.linear(input, weight, bias)
    where:
    - in_2 = input tensor
    - in_1 = weight tensor  
    - in_0 = bias tensor
    
    This is a fallback pattern that can match when more complex patterns
    (like dropout + linear) fail to match.
    """
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    return (linear,)


def replacement_args(in_0, in_1, in_2):
    """Extract arguments for the fused kernel.
    
    Pass the input, weight, and bias to the fused linear kernel.
    """
    return (in_2, in_1, in_0)


def replacement_func():
    from pass_dir.fused_linear_kernel import fused_linear
    return fused_linear