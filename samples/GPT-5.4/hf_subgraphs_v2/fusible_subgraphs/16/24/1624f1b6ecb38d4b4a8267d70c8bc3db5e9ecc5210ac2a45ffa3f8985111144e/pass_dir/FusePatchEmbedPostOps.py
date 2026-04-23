import torch

from pass_dir.vivit_shared import shared_replacement_func


def pattern(conv_out, cls_token, pos_embed):
    tmp_7 = conv_out.flatten(2)
    tmp_8 = tmp_7.transpose(1, 2)
    tmp_9 = cls_token.tile([1, 1, 1])
    tmp_10 = torch.cat((tmp_9, tmp_8), dim=1)
    tmp_11 = tmp_10 + pos_embed
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.0, False, False)
    return tmp_12


def replacement_args(conv_out, cls_token, pos_embed):
    return (conv_out, cls_token, pos_embed, "patch_embed_post")


def replacement_func():
    return shared_replacement_func()