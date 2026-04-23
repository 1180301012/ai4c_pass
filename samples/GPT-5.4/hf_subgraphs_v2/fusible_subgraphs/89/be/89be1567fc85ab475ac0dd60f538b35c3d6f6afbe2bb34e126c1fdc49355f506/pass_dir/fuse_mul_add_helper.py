import torch


def fused_mul_add_compiled(in_0, in_1, in_2):
    return torch.addcmul(in_0, in_2, in_1)