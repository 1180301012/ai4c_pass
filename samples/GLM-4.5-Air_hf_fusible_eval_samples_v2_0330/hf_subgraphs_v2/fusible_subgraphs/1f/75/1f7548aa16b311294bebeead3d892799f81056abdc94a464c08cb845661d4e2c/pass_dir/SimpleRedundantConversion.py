import torch

def pattern(x):
    # Very simple pattern with no dead code
    y = x.float()
    z = y.float()  # This is redundant - just convert twice
    return z

def replacement_args(x):
    # The pattern just takes one argument x
    return (x,)

def optimized_kernel(x):
    """
    Simple optimization: remove the redundant float conversion
    """
    # Just convert once instead of twice
    return x.float()

@torch.fx.wrap
def kernel_wrapper(x):
    """
    Simple wrapper that converts once instead of twice
    """
    return optimized_kernel(x)

def replacement_func():
    return kernel_wrapper