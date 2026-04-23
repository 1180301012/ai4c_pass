import torch
import triton
import triton.language as tl
from pass_dir.shared_scalar_attention_routes import replacement_func


def pattern(in_0, in_1, in_2):
    bmm = torch.bmm(in_0, in_1)
    tmp_1 = torch.nn.functional.softmax(bmm, dim=-1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p=0.0, training=False)
    bmm_1 = torch.bmm(tmp_2, in_2)
    tmp_4 = bmm_1.view(1, 16, 1, 64)
    tmp_5 = tmp_4.transpose(1, 2)
    tmp_6 = tmp_5.reshape(1, 1, 1024)
    return tmp_6


def replacement_args(in_0, in_1, in_2):
    return (in_2, "base_16_64")