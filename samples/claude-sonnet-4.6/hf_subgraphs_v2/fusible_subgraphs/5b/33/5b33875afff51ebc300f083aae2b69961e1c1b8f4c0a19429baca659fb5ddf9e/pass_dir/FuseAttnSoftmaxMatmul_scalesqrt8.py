import torch
from pass_dir.attn_shared import _universal


def pattern(a, b):
    """Matches: matmul(a, b).permute(0,2,1,3).contiguous()  for all graphs."""
    matmul = torch.matmul(a, b)
    tmp_0 = matmul.permute(0, 2, 1, 3)
    tmp_1 = tmp_0.contiguous()
    return tmp_1


def replacement_args(a, b):
    return (a, b)


def replacement_func():
    return _universal