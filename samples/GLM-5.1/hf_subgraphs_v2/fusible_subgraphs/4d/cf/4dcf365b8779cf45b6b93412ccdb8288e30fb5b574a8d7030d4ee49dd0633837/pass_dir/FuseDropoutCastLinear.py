import torch
from pass_dir._shared_kernel import fused_linear_dispatch

# Pattern: to(float16) + linear
# The dropout is a no-op (p=0, training=False), so we skip it in the pattern
# and just match the essential computation: cast + linear
def pattern(in_0, in_1, in_2):
    to = in_2.to(torch.float16)
    linear = torch.nn.functional.linear(to, in_1, in_0)
    return (linear,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_dropout_cast_linear")

def replacement_func():
    return fused_linear_dispatch