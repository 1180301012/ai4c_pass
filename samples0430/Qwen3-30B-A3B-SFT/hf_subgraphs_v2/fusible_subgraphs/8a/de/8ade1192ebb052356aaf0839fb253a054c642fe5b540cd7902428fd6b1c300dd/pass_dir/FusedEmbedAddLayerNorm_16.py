import torch
import triton
import triton.language as tl
from torch import device as torch_device
from pass_dir._kernels import _fused_embed_ln_dispatch  # noqa: F401


# ── Pattern & plumbing ────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4):
    tmp_9 = in_4.unsqueeze(0)
    tmp_10 = tmp_9 + 2
    tmp_11 = torch.nn.functional.embedding(tmp_10, in_1, None, None, 2.0, False, False)
    tmp_12 = tmp_11.to(torch_device(type='cuda', index=0))
    tmp_13 = in_0 + tmp_12
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (16,), in_3, in_2, 1e-05)
    return tmp_14


def replacement_args(in_0, in_1, in_2, in_3, in_4):
    return (in_0, in_1, in_2, in_3, in_4)


def replacement_func():
    return _fused_embed_ln_dispatch