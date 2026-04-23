import torch
from pass_dir._shared_kernel import fused_linear_dispatch

# Pattern: to(bfloat16) + linear
def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.bfloat16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return (linear,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_dropout_cast_linear_bf16")

def replacement_func():
    return fused_linear_dispatch