import torch
import triton
import triton.language as tl
from pass_dir.shared_patch_embed import patch_embed_ln_partition_dispatch, ROUTE_PATCH2_C16_WS2


def pattern(in_0, in_1, in_2, in_3, in_4):
    conv2d = torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1)
    tmp_6 = conv2d.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    tmp_10 = tmp_9.view(1, 16, 16, 16)
    tmp_11 = torch.nn.functional.pad(tmp_10, (0, 0, 0, 0, 0, 0), 'constant', None)
    tmp_12 = tmp_11.view(1, 8, 2, 8, 2, 16)
    tmp_13 = tmp_12.permute(0, 1, 3, 2, 4, 5)
    return (tmp_9, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (torch.conv2d(in_0, in_4, in_3, (2, 2), (0, 0), (1, 1), 1), in_1, in_2, ROUTE_PATCH2_C16_WS2)


def replacement_func():
    return patch_embed_ln_partition_dispatch