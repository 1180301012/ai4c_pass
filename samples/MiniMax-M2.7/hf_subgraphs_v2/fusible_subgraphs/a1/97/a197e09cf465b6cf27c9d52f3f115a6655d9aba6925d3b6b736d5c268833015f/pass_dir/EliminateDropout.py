import torch

@torch.fx.wrap
def identity(x: torch.Tensor) -> torch.Tensor:
    """
    Identity function - returns input unchanged.
    """
    return x


def pattern(in_0):
    """
    Match dropout with p=0 which is a no-op.
    """
    tmp_0 = torch.nn.functional.dropout(in_0, 0.0, False, False)
    return tmp_0


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return identity