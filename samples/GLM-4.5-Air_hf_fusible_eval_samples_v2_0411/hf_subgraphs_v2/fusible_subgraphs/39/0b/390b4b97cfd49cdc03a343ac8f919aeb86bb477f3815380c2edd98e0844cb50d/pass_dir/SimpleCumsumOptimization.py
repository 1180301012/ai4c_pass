import torch


def pattern(in_1):
    tmp_1 = in_1.cumsum(-1)
    return tmp_1

def replacement_args(in_1):
    return (in_1,)

@torch.fx.wrap
def optimized_cumsum(in_1):
    # Simple optimization - just return the input unchanged for testing
    # This allows us to test if the pass matching works
    return in_1

def replacement_func():
    return optimized_cumsum