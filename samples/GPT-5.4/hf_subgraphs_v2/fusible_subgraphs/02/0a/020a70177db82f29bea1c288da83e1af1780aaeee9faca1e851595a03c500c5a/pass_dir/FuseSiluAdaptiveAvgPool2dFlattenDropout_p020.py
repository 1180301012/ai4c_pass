import torch
from pass_dir.fused_silu_gap_shared import replacement_func


def pattern(in_0: torch.Tensor):
    tmp_0 = torch.nn.functional.silu(in_0, inplace=True)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, 1)
    tmp_2 = torch.flatten(tmp_1, 1)
    tmp_3 = torch.nn.functional.dropout(tmp_2, 0.2, False, True)
    return tmp_3


def replacement_args(in_0: torch.Tensor):
    return (in_0, "p020")