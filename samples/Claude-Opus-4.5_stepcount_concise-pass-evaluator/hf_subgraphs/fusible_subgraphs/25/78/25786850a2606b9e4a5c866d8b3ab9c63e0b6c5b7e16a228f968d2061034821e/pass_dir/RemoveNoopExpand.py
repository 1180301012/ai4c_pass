import torch

# Pattern: expand(1, -1, -1) on [1, 1, 768] tensor is a no-op
# The -1 means keep the same dimension, and first dim is already 1

def pattern(x):
    return x.expand(1, -1, -1)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def identity_expand(x):
    return x


def replacement_func():
    return identity_expand