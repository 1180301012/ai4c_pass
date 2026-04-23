import torch
import triton
import triton.language as tl
from pass_dir.shared_swin_patch_merge import replacement_func


def pattern(tmp_7, in_1, in_2):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (96,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(tmp_7, in_1, in_2):
    return (tmp_7, in_1, in_2, 'large_post_conv_c96')