import torch
import triton
import triton.language as tl
from pass_dir.attn_kernel import fused_dispatch


# ── Pattern: scale=8.0, reshape to [1, -1, 2, 64]  (Finnish-NLP/convbert) ──

def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = in_0 / 8.0
    tmp_1 = tmp_0 + in_2
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.1, False, False)
    matmul = torch.matmul(tmp_3, in_3)
    tmp_5 = matmul.permute(0, 2, 1, 3)
    tmp_6 = tmp_5.contiguous()
    tmp_7 = torch.reshape(in_1, [1, -1, 2, 64])
    return (tmp_6, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3, "scale8_2x64")


def replacement_func():
    return fused_dispatch