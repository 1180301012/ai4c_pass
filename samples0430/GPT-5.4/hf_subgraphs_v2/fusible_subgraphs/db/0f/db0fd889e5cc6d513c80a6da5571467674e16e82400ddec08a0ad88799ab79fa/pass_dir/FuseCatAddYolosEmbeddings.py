import torch
import triton
import triton.language as tl
from pass_dir.shared_yolos_routes import yolos_dispatch


def pattern(in_3, in_4, tmp_9, in_5):
    tmp_10 = in_3.expand(1, -1, -1)
    tmp_11 = in_4.expand(1, -1, -1)
    tmp_12 = torch.cat((tmp_10, tmp_9, tmp_11), dim=1)
    tmp_13 = in_5[(slice(None, None, None), 0, slice(None, None, None))]
    tmp_14 = tmp_13[(slice(None, None, None), None)]
    tmp_15 = in_5[(slice(None, None, None), slice(-10, None, None), slice(None, None, None))]
    tmp_16 = in_5[(slice(None, None, None), slice(1, -10, None), slice(None, None, None))]
    tmp_17 = tmp_16.transpose(1, 2)
    tmp_18 = tmp_17.view(1, 32, 15, 15)
    tmp_19 = torch.nn.functional.interpolate(tmp_18, size=(15, 15), mode='bicubic', align_corners=False)
    tmp_20 = tmp_19.flatten(2)
    tmp_21 = tmp_20.transpose(1, 2)
    tmp_22 = torch.cat((tmp_14, tmp_21, tmp_15), dim=1)
    tmp_23 = tmp_12 + tmp_22
    return tmp_23


def replacement_args(in_3, in_4, tmp_9, in_5):
    return (in_3, tmp_9, in_4, in_5, "cat_add")


def replacement_func():
    return yolos_dispatch