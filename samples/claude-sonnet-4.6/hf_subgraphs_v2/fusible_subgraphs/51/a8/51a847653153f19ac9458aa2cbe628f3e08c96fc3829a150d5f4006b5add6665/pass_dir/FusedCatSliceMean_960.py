import operator
import torch
from pass_dir.shared_cat_mean_kernel import triton_cat


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_0, in_1], dim=1)
    tmp_1 = operator.getitem(
        tmp_0,
        (slice(None, None, None), slice(None, 960, None),
         slice(None, None, None), slice(None, None, None))
    )
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return triton_cat