import torch

def pattern(x):
    tmp_2 = x.cos()
    tmp_3 = tmp_2 * 1.0
    tmp_4 = x.sin()
    tmp_5 = tmp_4 * 1.0
    tmp_6 = tmp_3.to(dtype=torch.bfloat16)
    tmp_7 = tmp_5.to(dtype=torch.bfloat16)
    return tmp_6, tmp_7

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_trigonometric_optimized(x):
    # Remove redundant multiplies by 1.0 and fuse type conversions
    cos_val = x.cos().to(torch.bfloat16)
    sin_val = x.sin().to(torch.bfloat16)
    return cos_val, sin_val

def replacement_func():
    return fused_trigonometric_optimized