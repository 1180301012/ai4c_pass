import torch


def optimized_mean(x: torch.Tensor) -> torch.Tensor:
    """
    Optimized mean reduction using PyTorch's optimized operations.
    Uses sum / count instead of native mean for better fusion opportunities.
    """
    # Compute sum and divide - this can sometimes be better optimized
    return x.sum(dim=-2, keepdim=True) / x.shape[-2]


def pattern(in_2):
    """
    Pattern: mean(dim=-2, keepdim=True)
    """
    tmp_4 = in_2.mean(dim=-2, keepdim=True)
    return tmp_4


def replacement_args(in_2):
    return (in_2,)


def replacement_func():
    return optimized_mean