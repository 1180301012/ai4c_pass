import torch
from pass_dir.shared_relu_flatten import replacement_func as _shared_replacement_func


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0, "relu_dropout_flatten")


def replacement_func():
    return _shared_replacement_func()