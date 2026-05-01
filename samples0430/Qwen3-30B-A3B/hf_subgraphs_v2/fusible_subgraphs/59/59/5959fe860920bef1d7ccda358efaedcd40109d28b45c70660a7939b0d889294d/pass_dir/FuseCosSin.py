import torch

def pattern(in_1):
    tmp_1 = torch.cat((in_1, in_1), dim=-1)
    tmp_2 = tmp_1.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = tmp_1.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(torch.bfloat16)
    tmp_7 = tmp_5.to(torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(in_1):
    return (in_1,)

@torch.fx.wrap
def kernel_wrapper(in_1):
    cat = torch.cat((in_1, in_1), dim=-1)
    cos_result = torch.cos(cat).to(torch.bfloat16)
    sin_result = torch.sin(cat).to(torch.bfloat16)
    return cos_result, sin_result

def replacement_func():
    return kernel_wrapper