import torch
from pass_dir.shared_relu_add_gap import replacement_func


def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.relu(in_1, inplace=False)
    tmp_1 = tmp_0 + in_0
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1, "relu_add")