import torch
from pass_dir.shared_kernel import replacement_func


def pattern(conv2d_out, in_1, in_2):
    tmp_6 = conv2d_out.flatten(2)
    tmp_7 = tmp_6.transpose(1, 2)
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (16,), in_2, in_1, 1e-05)
    tmp_9 = torch.nn.functional.dropout(tmp_8, 0.0, False, False)
    return tmp_9


def replacement_args(conv2d_out, in_1, in_2):
    # in_2 is norm_weight, in_1 is norm_bias (matching layer_norm order)
    return (conv2d_out, in_2, in_1, "tiny")