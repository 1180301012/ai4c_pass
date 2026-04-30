import torch
from torch import device
from pass_dir.shared_dispatch import dispatch_wrapper


def pattern(size):
    result = torch.ones((size,), dtype=torch.float32, device=device(type='cuda'))
    return result


def replacement_args(size):
    return (size, "ones")


def replacement_func():
    return dispatch_wrapper