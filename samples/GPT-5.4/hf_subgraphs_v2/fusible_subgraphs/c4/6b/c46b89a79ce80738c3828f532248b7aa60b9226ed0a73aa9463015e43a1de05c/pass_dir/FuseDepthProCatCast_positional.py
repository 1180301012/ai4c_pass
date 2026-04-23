import torch

from pass_dir.shared_depthpro_dispatch import depthpro_dispatch


def pattern(a, b, c):
    tmp_0 = torch.cat([a, b, c], 0)
    tmp_1 = tmp_0.to(torch.float16)
    return tmp_1


def replacement_args(a, b, c):
    return (a, b, c, "cat_cast")


def replacement_func():
    return depthpro_dispatch