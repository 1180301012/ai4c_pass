import torch
from pass_dir.depthpro_shared import replacement_func


def pattern(tmp_5, tmp_2, in_0):
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim=0)
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7


def replacement_args(tmp_5, tmp_2, in_0):
    return (tmp_5, tmp_2, in_0, "cat")