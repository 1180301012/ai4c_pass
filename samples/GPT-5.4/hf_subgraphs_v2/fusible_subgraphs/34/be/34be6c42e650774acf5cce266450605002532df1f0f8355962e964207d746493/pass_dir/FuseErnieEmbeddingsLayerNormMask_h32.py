import torch
from torch import device

from pass_dir.shared_ernie_embeddings import ernie_embeddings_dispatch


def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_5 = in_0.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38
    tmp_7 = tmp_6
    tmp_8 = tmp_7.unsqueeze(1)
    tmp_9 = tmp_8.unsqueeze(1)
    tmp_10 = torch.nn.functional.embedding(in_0, in_4, 1, None, 2.0, False, False)
    tmp_11 = torch.ones((1, 15), dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_12 = torch.cumsum(tmp_11, dim=1)
    tmp_13 = tmp_12 - tmp_11
    tmp_13 += 2
    tmp_14 = tmp_13
    tmp_15 = torch.nn.functional.embedding(tmp_14, in_3, 1, None, 2.0, False, False)
    tmp_16 = tmp_10 + tmp_15
    tmp_17 = torch.nn.functional.layer_norm(tmp_16, (32,), in_2, in_1, 1e-05)
    tmp_18 = torch.nn.functional.dropout(tmp_17, 0.1, False, False)
    return (tmp_18, tmp_9)


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4, "h32")


def replacement_func():
    return ernie_embeddings_dispatch