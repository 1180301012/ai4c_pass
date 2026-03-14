import torch
import triton
import triton.language as tl

# Pattern: avgpool + flatten (equivalent to mean over spatial dims)
def pattern(x):
    tmp_6 = torch.nn.functional.adaptive_avg_pool2d(x, 1)
    tmp_7 = tmp_6.flatten(1, -1)
    return tmp_7

def replacement_args(x):
    return (x,)

@torch.fx.wrap
def fused_avgpool_flatten(x):
    # Use PyTorch's efficient mean - equivalent to adaptive_avg_pool2d(x, 1).flatten(1, -1)
    return x.mean(dim=(2, 3))

def replacement_func():
    return fused_avgpool_flatten