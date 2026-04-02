import torch

def pattern(in_0, in_1, in_2):
    tmp_0 = in_1 @ in_0
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_2 = in_2[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    tmp_3 = tmp_2.transpose(-1, -2)
    tmp_4 = tmp_3.reshape(1, 256, 14, 14)
    tmp_5 = torch.functional.split(tmp_4, [64, 96, 96], dim=1)
    tmp_6 = tmp_5[0]
    tmp_7 = tmp_5[1]
    tmp_8 = tmp_5[2]
    return tmp_0, tmp_6, tmp_7, tmp_8, tmp_1

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def _fused_256_14_14(in_0, in_1, in_2):
    matmul = in_1 @ in_0
    tmp_1 = in_1[:, :, 1:, :]
    tmp_v = in_2[:, :, 1:, :].permute(0, 1, 3, 2).contiguous()
    tmp_4 = tmp_v.reshape(1, 256, 14, 14)
    s0 = tmp_4[:, :64, :, :]
    s1 = tmp_4[:, 64:160, :, :]
    s2 = tmp_4[:, 160:, :, :]
    return matmul, s0, s1, s2, tmp_1

def replacement_func():
    return _fused_256_14_14