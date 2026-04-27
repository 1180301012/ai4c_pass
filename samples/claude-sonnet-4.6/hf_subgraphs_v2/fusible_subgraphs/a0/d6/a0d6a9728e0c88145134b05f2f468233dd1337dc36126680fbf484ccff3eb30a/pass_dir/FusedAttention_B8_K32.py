import torch
from pass_dir._shared_attn_kernel import fused_attn_dispatch


def pattern(attn_weights, in_2):
    # Match: bmm([B,1,1],[B,1,32]) + view(1,8,1,32) + transpose(1,2) + reshape(1,1,256)
    # attn_weights is the dropout output (tmp_2); we skip matching bmm1/softmax/dropout
    bmm_1 = torch.bmm(attn_weights, in_2)
    tmp_4 = bmm_1.view(1, 8, 1, 32)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 256)
    return tmp_6


def replacement_args(attn_weights, in_2):
    return (attn_weights, in_2, "b8k32")


def replacement_func():
    return fused_attn_dispatch