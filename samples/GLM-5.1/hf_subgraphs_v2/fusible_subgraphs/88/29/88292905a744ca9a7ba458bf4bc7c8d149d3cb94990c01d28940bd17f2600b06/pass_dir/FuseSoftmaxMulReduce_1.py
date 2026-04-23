import torch
from pass_dir.shared_dispatch import dispatch_wrapper


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.softmax(in_1, dim=1)
    tmp_1 = tmp_0.reshape(1, -1)
    tmp_2 = tmp_1.view(1, -1, 1, 1)
    tmp_3 = tmp_2.view(1, 2, -1, 1, 1)
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    tmp_6 = tmp_5.contiguous()
    return (tmp_6,)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "route_1")


def replacement_func():
    return dispatch_wrapper