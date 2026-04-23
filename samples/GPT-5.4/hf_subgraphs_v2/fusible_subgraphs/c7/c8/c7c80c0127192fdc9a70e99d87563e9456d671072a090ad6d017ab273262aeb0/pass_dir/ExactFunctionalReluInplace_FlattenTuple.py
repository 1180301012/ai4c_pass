import torch
import graph_net_bench.torch.custom_replacement as _custom_replacement
from pass_dir.relu_shared import relu_dispatch

# Preserve kwargs in pattern tracing so the pattern matches the real FX graph,
# which contains torch.nn.functional.relu(..., inplace=True) with kwargs.
_custom_replacement.force_args_symbolic_trace = torch.fx.symbolic_trace


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=True)
    tmp_1 = tmp_0.flatten(1, -1)
    return tmp_1


def replacement_args(in_0):
    return (in_0, "inplace")


def replacement_func():
    return relu_dispatch