import torch
from pass_dir.shared_fused_gelu_pad import replacement_func


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0, approximate='none')
    tmp_1 = tmp_0.reshape(1, 124, 2, 768)
    tmp_2 = tmp_1.reshape(1, 248, 768)
    tmp_3 = torch.nn.functional.pad(tmp_2, (0, 0, 0, 1), 'constant', None)
    return tmp_3


def replacement_args(in_0):
    return (in_0, 'gelu_approx_none_reshape_pad')