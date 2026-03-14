import torch


# Pattern matching function - match just add (silu_output + in_0)
def pattern(in_0, silu_output):
    """
    Match the Add pattern where silu_output is already computed
    """
    result = silu_output + in_0
    return result


# Extract arguments needed for replacement
def replacement_args(in_0, silu_output):
    return (in_0, silu_output)


# Wrapper function - use native torch operations
@torch.fx.wrap
def native_add(in_0, silu_output):
    return silu_output + in_0


# Replacement function - returns the wrapper function
def replacement_func():
    return native_add