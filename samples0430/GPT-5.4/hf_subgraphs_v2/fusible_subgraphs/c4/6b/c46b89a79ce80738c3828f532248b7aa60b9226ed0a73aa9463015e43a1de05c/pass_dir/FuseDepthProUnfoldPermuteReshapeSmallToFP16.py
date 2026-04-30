import torch
from pass_dir.depthpro_shared import replacement_func


def pattern(in_1):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size=(384, 384), stride=(192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    return tmp_2


def replacement_args(in_1):
    return (in_1, "small")