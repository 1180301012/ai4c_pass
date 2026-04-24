import torch
import triton
import triton.language as tl
from pass_dir.shared_kernel import fused_view_cat_sigmoid_sub_mul


def pattern(conv_out, in_3, in_4):
    tmp_3 = conv_out.view(1, 1, -1)
    tmp_4 = torch.cat([in_3, in_4, tmp_3], 2)
    tmp_5 = tmp_4.sigmoid()
    tmp_6 = tmp_5 - 0.25
    tmp_7 = tmp_6 * 3.141592653589793
    return tmp_7


def replacement_args(conv_out, in_3, in_4):
    # in_3: [1, 1, 6400], in_4: [1, 1, 1600]
    # conv_out: [1, 1, 20, 20] → conv_last = 400
    L1 = 6400
    L2 = 1600
    conv_last = 400
    return (conv_out, in_3, in_4, L1, L2, conv_last)


def replacement_func():
    return fused_view_cat_sigmoid_sub_mul