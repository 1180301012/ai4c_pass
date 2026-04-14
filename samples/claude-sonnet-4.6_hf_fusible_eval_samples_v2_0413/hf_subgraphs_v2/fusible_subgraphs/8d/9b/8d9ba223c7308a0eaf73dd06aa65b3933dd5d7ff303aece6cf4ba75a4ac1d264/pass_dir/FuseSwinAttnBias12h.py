import torch
from pass_dir.swin_shared import fused_swin_dispatch


# Extended pattern: 16*sigmoid_out + in_2 + 2*in_3 (covers scale + all 3 adds + views)
# sigmoid_out (tmp9) is the aten.sigmoid output treated as wildcard input.
def pattern(tmp9, in_2, in_3):
    tmp_10 = 16 * tmp9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    return tmp_19


def replacement_args(tmp9, in_2, in_3):
    return (tmp9, in_2, in_3, "12h")


def replacement_func():
    return fused_swin_dispatch