import torch
from pass_dir.bn_kernels import bn_slice_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5[slice(None, None, None), slice(256, None, None), slice(None, None, None), slice(None, None, None)]
    tmp_5 = torch.nn.functional.batch_norm(in_4, in_0, in_1, in_3, in_2, False, 0.1, 0.001)
    return (tmp_5, tmp_4)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5, 256)


def replacement_func():
    return bn_slice_dispatch