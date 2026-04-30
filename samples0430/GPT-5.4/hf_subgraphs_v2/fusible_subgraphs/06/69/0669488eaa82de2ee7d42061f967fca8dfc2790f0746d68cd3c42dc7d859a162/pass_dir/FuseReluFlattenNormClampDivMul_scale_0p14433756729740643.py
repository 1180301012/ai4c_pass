import torch
from pass_dir.rtmw_norm_relu_mul_shared import rtmw_relu_rmsnorm_mul


def pattern(in_0, in_1):
    tmp_1 = torch.nn.functional.relu(in_1, inplace = True)
    tmp_2 = torch.flatten(tmp_1, 2)
    tmp_3 = torch.functional.norm(tmp_2, dim = -1, keepdim = True)
    tmp_4 = tmp_3 * 0.14433756729740643
    tmp_5 = tmp_4.clamp(min = 1e-05)
    tmp_6 = tmp_2 / tmp_5
    tmp_7 = tmp_6 * in_0
    return (tmp_7,)


def replacement_args(in_0, in_1):
    return (in_1, in_0, 'scale_0p14433756729740643')


def replacement_func():
    return rtmw_relu_rmsnorm_mul