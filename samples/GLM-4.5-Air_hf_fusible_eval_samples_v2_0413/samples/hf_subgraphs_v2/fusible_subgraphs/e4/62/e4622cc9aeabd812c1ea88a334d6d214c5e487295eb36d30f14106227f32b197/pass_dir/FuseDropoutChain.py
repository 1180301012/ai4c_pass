import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """
    Match chain of two dropout operations with p=0.0.
    Instead of eliminating them separately, we can fuse them into a single identity operation.
    This reduces overhead from multiple function calls.
    """
    # First dropout
    tmp_1 = torch.nn.functional.dropout(input_tensor, 0.0, False, False)
    # Second dropout chained with None assignment
    result = torch.nn.functional.dropout(tmp_1, 0.0, False, False)
    return result

def replacement_args(input_tensor):
    """Return the arguments needed for replacement"""
    return (input_tensor,)

def fused_dropout_replacement(input_tensor):
    """
    Fused replacement for chain of two zero-probability dropouts.
    Since both are no-ops, this is equivalent to a single identity operation.
    This reduces function call overhead compared to two separate replacements.
    """
    return input_tensor

def replacement_func():
    """Return the fused replacement function"""
    return fused_dropout_replacement