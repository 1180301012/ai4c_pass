import torch
from pass_dir.shared_fused_cat_nearest_interpolate_stack40 import replacement_func


def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat((in_2, in_3), 1)
    tmp_1 = torch.nn.functional.interpolate(in_0, (40, 40), None, 'nearest', None, None, False)
    tmp_2 = torch.nn.functional.interpolate(in_1, (40, 40), None, 'nearest', None, None, False)
    tmp_3 = torch.stack([tmp_1, tmp_2, tmp_0])
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)