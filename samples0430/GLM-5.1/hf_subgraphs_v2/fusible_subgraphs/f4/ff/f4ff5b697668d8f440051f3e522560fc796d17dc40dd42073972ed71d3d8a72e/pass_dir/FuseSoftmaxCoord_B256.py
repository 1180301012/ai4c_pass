import torch
from pass_dir.fused_kernel import fused_softmax_coord


def pattern(in_0, in_1, in_2):
    """
    Pattern for batch size B=256.
    Matches: softmax -> reshape(-1,17,64,64) -> mul(in_0) -> reshape(256,17,-1) -> sum -> mul(in_1) -> reshape(256,17,-1) -> sum -> cat
    """
    tmp_2 = torch.nn.functional.softmax(in_2, dim=2)
    tmp_3 = tmp_2.reshape(-1, 17, 64, 64)
    tmp_4 = tmp_3.mul(in_0)
    tmp_5 = tmp_4.reshape(256, 17, -1)
    tmp_6 = torch.sum(tmp_5, dim=2, keepdim=True)
    tmp_7 = tmp_3.mul(in_1)
    tmp_8 = tmp_7.reshape(256, 17, -1)
    tmp_9 = torch.sum(tmp_8, dim=2, keepdim=True)
    tmp_10 = torch.cat([tmp_6, tmp_9], dim=-1)
    return (tmp_3, tmp_10)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_B256")


def replacement_func():
    return fused_softmax_coord