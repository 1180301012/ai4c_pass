import torch

def pattern(x):
    tmp_5 = x.__eq__(1)
    tmp_6 = tmp_5.to(torch.float32)
    tmp_6 *= -3.4028234663852886e+38
    tmp_7 = tmp_6.unsqueeze(1)
    tmp_8 = tmp_7.unsqueeze(1)
    return tmp_8

def replacement_args(x):
    return (x,)

def replacement_func():
    return lambda x: x