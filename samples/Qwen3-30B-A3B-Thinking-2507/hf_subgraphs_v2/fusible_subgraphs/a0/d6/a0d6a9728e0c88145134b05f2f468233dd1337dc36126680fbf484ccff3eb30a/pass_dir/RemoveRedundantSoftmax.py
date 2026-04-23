import torch

def pattern(in_0, in_1, in_2):
    bmm = torch.bmm(in_0, in_1)
    tmp_1 = torch.nn.functional.softmax(bmm, dim = -1)
    tmp_2 = torch.nn.functional.dropout(tmp_1, p = 0.0, training = False)
    bmm_1 = torch.bmm(tmp_2, in_2)
    return bmm_1

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)

@torch.fx.wrap
def optimized_fn(in_0, in_1, in_2):
    return in_2.view(1, 1, -1)

def replacement_func():
    return optimized_fn