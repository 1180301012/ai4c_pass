import torch

# Pattern: dropout(x, 0.0, False, False) is a no-op
# Remove the overhead by replacing with identity

def pattern(x):
    return torch.nn.functional.dropout(x, 0.0, False, False)


def replacement_args(x):
    return (x,)


@torch.fx.wrap  
def passthrough(x):
    return x


def replacement_func():
    return passthrough