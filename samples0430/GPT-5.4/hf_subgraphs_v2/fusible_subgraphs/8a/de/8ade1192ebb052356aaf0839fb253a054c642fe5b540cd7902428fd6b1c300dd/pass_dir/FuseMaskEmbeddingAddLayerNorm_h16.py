import torch
from torch import device
import triton
import triton.language as tl

from pass_dir.shared_impl import fused_mask_embedding_add_layernorm


def pattern(in_0: torch.Tensor, in_1, in_2, in_3, in_4, in_5):
    tmp_4 = in_5.to(torch.float32)
    tmp_5 = torch.tensor(1.0, dtype=torch.float32)
    tmp_6 = tmp_5 - tmp_4
    tmp_7 = tmp_6.to(torch.bool)
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (16,), in_3, in_2, 1e-05)
    tmp_15 = torch.nn.functional.dropout(tmp_14, p=0.1, training=False)
    return (tmp_8, tmp_15)


def replacement_args(in_0: torch.Tensor, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_mask_embedding_add_layernorm