"""
Full-model fusion pass for H=2048 (float16 model).

Pattern mirrors model.py EXACTLY (same structure as H=1024 but view/LN use 2048).
"""
import torch
import triton          # noqa: F401
import triton.language as tl  # noqa: F401
from torch import device

from pass_dir.shared_dispatch import shared_fused_add_ln


def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.arange(0, 9, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_5 += 2
    tmp_7 = tmp_5.view(-1)
    tmp_8 = in_1.index_select(0, tmp_7)
    tmp_9 = tmp_8.view(1, 9, 2048)
    tmp_10 = tmp_9.detach()
    tmp_11 = tmp_10.to(device(type='cuda', index=0))
    tmp_12 = in_0 + tmp_11
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (2048,), in_3, in_2, 1e-05)
    return (tmp_13, tmp_14)


def replacement_args(in_0, in_1, in_2, in_3):
    # a=in_0, b=in_1, c=in_2(bias), d=in_3(weight)
    return (in_0, in_1, in_2, in_3, "full_2048")


def replacement_func():
    return shared_fused_add_ln