import torch
from pass_dir._shared_kernel import fused_linear_dispatch

# Pattern: dropout(training=False) + linear
# Since dropout with training=False is a no-op, this is effectively just linear
def pattern(in_0, in_1, in_2):
    tmp_3 = torch.nn.functional.dropout(in_2, 0.1, False, False)
    linear = torch.nn.functional.linear(tmp_3, in_1, in_0)
    return (linear,)

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "route_dropout_linear")

def replacement_func():
    return fused_linear_dispatch