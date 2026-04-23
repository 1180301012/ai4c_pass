import torch
from pass_dir._shared_kernel import flash_attention_dispatch

def pattern(in_0, in_1, in_2):
    tmp_0 = torch.matmul(in_0, in_1)
    tmp_1 = tmp_0 / 8.0
    tmp_2 = torch.nn.functional.softmax(tmp_1, dim=-1)
    tmp_4 = torch.matmul(tmp_2, in_2)
    tmp_5 = tmp_4.permute(0, 2, 1, 3)
    return (tmp_5,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_8_p0")

def replacement_func():
    return flash_attention_dispatch