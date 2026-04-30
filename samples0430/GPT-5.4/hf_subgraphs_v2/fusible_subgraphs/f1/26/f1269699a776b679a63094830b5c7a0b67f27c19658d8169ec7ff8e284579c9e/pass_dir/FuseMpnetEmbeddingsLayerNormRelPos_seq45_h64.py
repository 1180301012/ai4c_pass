import torch
from pass_dir.mpnet_relpos_const_common import replacement_func


def pattern(x):
    tmp = torch.nn.functional.dropout(x, 0.1, False, False)
    return tmp


def replacement_args(x):
    return (x, "dropout_identity")