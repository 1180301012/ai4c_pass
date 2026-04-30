import torch
from pass_dir.shared_dispatch import replacement_func


def pattern(in_0, in_1, in_2, tmp_2):
    tmp_3 = tmp_2.view(-1, 32, 32, 768)
    tmp_4 = torch.roll(tmp_3, shifts = (4, 4), dims = (1, 2))
    tmp_5 = tmp_4.view(1, 1024, 768)
    tmp_6 = torch.nn.functional.layer_norm(tmp_5, (768,), in_1, in_0, 1e-05)
    tmp_7 = in_2 + tmp_6
    return tmp_7


def replacement_args(in_0, in_1, in_2, tmp_2):
    return (in_0, in_1, in_2, tmp_2, "route_32_32_768")