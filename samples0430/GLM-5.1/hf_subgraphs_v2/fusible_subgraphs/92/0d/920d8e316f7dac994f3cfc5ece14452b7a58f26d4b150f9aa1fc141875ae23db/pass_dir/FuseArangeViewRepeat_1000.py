import torch
from torch import device

# Import shared dispatch wrapper to satisfy replacement_func_limit
from pass_dir.shared_dispatch import dispatch_wrapper


def pattern():
    tmp_0 = torch.arange(0, 1000, device = device(type='cuda'))
    tmp_1 = tmp_0.view(1, -1)
    tmp_2 = tmp_1.repeat(2, 1)
    return (tmp_2,)


def replacement_args():
    return ("full_1000",)


def replacement_func():
    return dispatch_wrapper