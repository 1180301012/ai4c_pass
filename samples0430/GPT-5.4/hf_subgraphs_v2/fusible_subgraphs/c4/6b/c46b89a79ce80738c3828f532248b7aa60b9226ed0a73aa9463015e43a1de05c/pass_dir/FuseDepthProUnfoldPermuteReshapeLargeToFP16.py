import torch
from pass_dir.depthpro_shared import replacement_func


def pattern(in_2):
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size=(384, 384), stride=(288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    return tmp_5


def replacement_args(in_2):
    return (in_2, "large")