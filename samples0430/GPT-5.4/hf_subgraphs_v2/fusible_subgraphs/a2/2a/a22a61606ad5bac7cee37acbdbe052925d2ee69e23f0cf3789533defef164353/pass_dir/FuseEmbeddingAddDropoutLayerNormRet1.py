import torch
from pass_dir.embedding_add_layernorm_shared import dispatch_embedding_add_layernorm


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    tmp_7 = torch.nn.functional.embedding(in_0, in_5, 0, None, 2.0, False, False)
    tmp_8 = torch.nn.functional.embedding(in_6, in_4, None, None, 2.0, False, False)
    tmp_9 = tmp_7 + tmp_8
    tmp_10 = torch.nn.functional.embedding(in_7, in_3, None, None, 2.0, False, False)
    tmp_9 += tmp_10
    tmp_11 = tmp_9
    tmp_12 = torch.nn.functional.dropout(tmp_11, 0.1, False, False)
    tmp_13 = torch.nn.functional.layer_norm(tmp_12, (in_2.shape[0],), in_2, in_1, 1e-12)
    return (tmp_13,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_0, in_5, in_4, in_3, in_2, in_1, in_6, in_7, "ret1")


def replacement_func():
    return dispatch_embedding_add_layernorm