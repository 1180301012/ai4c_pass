import torch
from pass_dir._rope_shared import rope_dispatch


def pattern(in_0, in_4, dtype_tensor):
    tensor_split = in_0.tensor_split(2, -1)
    tmp_14 = tensor_split[0]
    tmp_15 = tensor_split[1]
    tmp_11 = in_4[(slice(None, None, None), slice(None, None, None), slice(None, 1, None), slice(None, None, None))]
    tmp_12 = in_4[(slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None))]
    tmp_16 = tmp_12 * tmp_15
    tmp_17 = tmp_12[(Ellipsis, slice(1, None, 2))]
    tmp_18 = -tmp_17
    tmp_19 = tmp_12[(Ellipsis, slice(None, None, 2))]
    tmp_20 = torch.stack([tmp_18, tmp_19], -1)
    tmp_21 = tmp_20.reshape((1, 12, 196, 64))
    tmp_22 = tmp_21 * tmp_14
    tmp_23 = tmp_16 + tmp_22
    tmp_24 = torch.cat([tmp_11, tmp_23], dim=2)
    tmp_25 = tmp_24.type_as(dtype_tensor)
    return tmp_25


def replacement_args(in_0, in_4, dtype_tensor):
    return (in_0, in_4, dtype_tensor, "split_12_196")


def replacement_func():
    return rope_dispatch