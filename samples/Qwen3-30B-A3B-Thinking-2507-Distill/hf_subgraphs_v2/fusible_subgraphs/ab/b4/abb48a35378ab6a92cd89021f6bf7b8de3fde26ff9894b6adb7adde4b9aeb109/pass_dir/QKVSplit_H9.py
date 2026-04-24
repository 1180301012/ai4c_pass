"""
Pass: FuseLinearQKV_H9
Fuses: linear -> reshape(1,197,3,9,48) -> permute(2,0,3,1,4) -> unbind(0) -> getitem x3 -> transpose(-2,-1)
for the case H=9 (convit_small, float32 and bfloat16 variants).
"""
import torch
from pass_dir.qkv_fused_kernel import _qkv_dispatch


def pattern(in_0, in_1):
    linear = torch.nn.functional.linear(in_1, in_0, None)
    tmp_2 = linear.reshape(1, 197, 3, 9, 48)
    tmp_3 = tmp_2.permute(2, 0, 3, 1, 4)
    unbind = tmp_3.unbind(0)
    tmp_5 = unbind[0]
    tmp_6 = unbind[1]
    tmp_7 = unbind[2]
    tmp_8 = tmp_6.transpose(-2, -1)
    return (tmp_5, tmp_8, tmp_7)


def replacement_args(in_0, in_1):
    return (in_0, in_1, "H9")


def replacement_func():
    return _qkv_dispatch