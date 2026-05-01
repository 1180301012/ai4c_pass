import torch
import os
import sys

from torch import device

# Make pass_dir importable as a package so we can share the dispatch wrapper
_pass_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_pass_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from pass_dir.shared_kernels import fused_add_layernorm_dispatch  # noqa: E402


def pattern(a, b, w, bias):
    tmp = a + b
    out = torch.nn.functional.layer_norm(tmp, (32,), w, bias, 1e-05)
    out2 = torch.nn.functional.dropout(out, 0.1, False, False)
    return out2


def replacement_args(a, b, w, bias):
    return (a, b, w, bias, "route_32")


def replacement_func():
    return fused_add_layernorm_dispatch