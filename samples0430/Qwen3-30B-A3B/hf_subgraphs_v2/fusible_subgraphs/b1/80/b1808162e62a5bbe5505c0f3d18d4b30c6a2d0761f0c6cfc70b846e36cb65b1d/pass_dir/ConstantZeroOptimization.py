import torch

def pattern(in_0):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = 1.0 - tmp_0
    tmp_2 = tmp_1.bool()
    tmp_3 = tmp_1.masked_fill(tmp_2, -3.4028234663852886e+38)
    tmp_4 = tmp_3 * tmp_1
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

def optimized_zero(in_0):
    return torch.zeros_like(in_0.to(torch.float32))
def replacement_func():
    return optimized_zero