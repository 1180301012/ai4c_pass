import torch


def pattern(in_3, in_1, in_0):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = linear.view(1, -1, 2, 64)
    tmp_4 = tmp_3.transpose(1, 2)
    return tmp_4

def replacement_args(in_3, in_1, in_0):
    return (in_3, in_1, in_0)

def replacement_func():
    def fused_linear(in_3, in_1, in_0):
        linear_val = torch.nn.functional.linear(in_3, in_1, in_0)
        out = linear_val.view(1, 2, -1, 64)
        return out
    return fused_linear