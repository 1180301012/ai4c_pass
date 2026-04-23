import torch

from pass_dir.shared_depthpro_dispatch import depthpro_dispatch


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.unfold(in_1, (384, 384), 1, 0, (192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, (384, 384), 1, 0, (288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], 0)
    tmp_7 = tmp_6.to(torch.float16)
    return tmp_7


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "full_pack")


def replacement_func():
    return depthpro_dispatch