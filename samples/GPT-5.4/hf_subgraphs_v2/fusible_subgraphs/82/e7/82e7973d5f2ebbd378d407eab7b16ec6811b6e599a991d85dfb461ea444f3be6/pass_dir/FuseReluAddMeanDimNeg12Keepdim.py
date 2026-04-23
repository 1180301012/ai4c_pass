import torch

from pass_dir.shared_fused_relu_add_gap import replacement_func


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    tmp_2 = torch.ops.aten.mean.dim(tmp_1, [-1, -2], True)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1, "relu_add_mean_neg12")