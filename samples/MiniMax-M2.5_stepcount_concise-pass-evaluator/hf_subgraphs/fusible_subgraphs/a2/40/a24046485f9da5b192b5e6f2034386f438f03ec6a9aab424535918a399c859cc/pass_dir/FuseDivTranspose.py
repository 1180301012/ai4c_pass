import torch

# The constant used in the original computation
DIV_CONSTANT = 1.6817928305074292


def div_transpose_optimal(x: torch.Tensor) -> torch.Tensor:
    """
    Optimal implementation using standard PyTorch operations.
    PyTorch's native transpose is a view operation (no memory copy).
    """
    # Direct operations - PyTorch will optimize these
    return (x / DIV_CONSTANT).transpose(-1, -2)


@torch.fx.wrap
def fuse_div_transpose_wrapper(x: torch.Tensor) -> torch.Tensor:
    """Wrapper function that will replace the original pattern."""
    return div_transpose_optimal(x)


# Pattern matching function
def pattern(in_0):
    """ 
    Match the pattern: division by constant followed by transpose of last two dims.
    
    Original computation:
        tmp_0 = in_0 / 1.6817928305074292
        tmp_1 = tmp_0.transpose(-1, -2)
    """
    tmp_0 = in_0 / DIV_CONSTANT
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


# Argument extraction function
def replacement_args(in_0):
    # Extract and return arguments needed for the replacement
    return (in_0,)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fuse_div_transpose_wrapper