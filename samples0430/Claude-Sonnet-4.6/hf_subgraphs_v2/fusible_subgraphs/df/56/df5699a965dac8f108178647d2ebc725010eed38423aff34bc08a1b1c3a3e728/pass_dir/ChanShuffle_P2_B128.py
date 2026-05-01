import torch
from pass_dir.shared_kernels import dispatch


def pattern(A, B_in):
    tmp  = torch.cat([A, B_in], dim=1)
    tmp1 = tmp.view(128, 2, 40, 32, 24)
    tmp2 = torch.transpose(tmp1, 1, 2)
    tmp3 = tmp2.contiguous()
    tmp4 = tmp3.view(128, 80, 32, 24)
    chunks = tmp4.chunk(2, dim=1)
    out1 = chunks[0]
    out2 = chunks[1]
    return out1, out2


def replacement_args(A, B_in):
    return (A, B_in, A, A, "sh")


def replacement_func():
    return dispatch