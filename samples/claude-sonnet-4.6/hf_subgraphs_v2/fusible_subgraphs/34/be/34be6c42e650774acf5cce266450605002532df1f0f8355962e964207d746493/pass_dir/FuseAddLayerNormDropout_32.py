import torch
import triton
from pass_dir.ernie_emb_kernels import ernie_dispatch


# Full-fusion pattern: word_embedding + position_embedding + add + LayerNorm(32) + dropout
# `pos_indices` is a FREE VARIABLE matching the intermediate position-index tensor.
def pattern(in_0, in_4, pos_indices, in_3, in_2, in_1):
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_15 = torch.nn.functional.embedding(pos_indices, in_3, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return tmp_18


def replacement_args(in_0, in_4, pos_indices, in_3, in_2, in_1):
    return (in_0, in_4, in_3, in_2, in_1, "32")


def replacement_func():
    return ernie_dispatch