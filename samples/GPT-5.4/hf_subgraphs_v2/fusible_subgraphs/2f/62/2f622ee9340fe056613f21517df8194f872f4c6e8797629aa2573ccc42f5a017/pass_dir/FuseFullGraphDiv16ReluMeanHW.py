import torch
from pass_dir.shared_fused_inplace_relu_mean import replacement_func


def pattern(in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=True)
    tmp_3 = torch.mean(tmp_0, (2, 3), True)
    return (tmp_0, tmp_3)


def replacement_args(in_1):
    return (in_1,)