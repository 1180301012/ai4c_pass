import torch

def pattern(tmp_6):
    tmp_7 = tmp_6.to(dtype=torch.float16)
    return tmp_7

def replacement_args(tmp_6):
    return (tmp_6,)

@torch.fx.wrap
def type_conversion_optimized(tmp_6):
    return tmp_6.to(dtype=torch.float16)

def replacement_func():
    return type_conversion_optimized