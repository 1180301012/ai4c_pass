import torch


# Pattern matching: unsqueeze followed by expand(1, -1)
# The expand(1, -1) is a no-op when the tensor already has size 1 on dim 0
def pattern(tmp_0):
    tmp_1 = tmp_0.unsqueeze(0)
    tmp_2 = tmp_1.expand(1, -1)
    return tmp_2


def replacement_args(tmp_0):
    return (tmp_0,)


# Use view instead of unsqueeze - view is a metadata operation with zero copy
# This is the most lightweight replacement possible
@torch.fx.wrap
def kernel_wrapper(tmp_0):
    # Use view to create [1, n] shape - no data copy, just metadata
    # Equivalent to unsqueeze(0) but expressed as view
    return tmp_0.view(1, -1)


def replacement_func():
    return kernel_wrapper