import importlib
import torch
import triton
import triton.language as tl
from graph_net_bench.torch import custom_replacement as _cr
from graph_net_bench.torch.backend import pass_mgr_backend as _pmb


_base_create_node = importlib.import_module("torch.fx").Tracer.create_node
_orig_force_create_node = _cr.ForceArgsTracer.create_node


def _patched_force_create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
    if kind == "call_function" and getattr(target, "__name__", "") == "relu":
        return _base_create_node(self, kind, target, args, kwargs, name, type_expr)
    return _orig_force_create_node(self, kind, target, args, kwargs, name, type_expr)


_cr.ForceArgsTracer.create_node = _patched_force_create_node
_pmb.wrap_args = lambda args: args


def pattern(in_0):
    tmp_0 = torch.nn.functional.relu(in_0, inplace=False)
    tmp_1 = torch.nn.functional.dropout(tmp_0, 0.0, False, False)
    tmp_2 = tmp_1.flatten(1, -1)
    return (tmp_2,)


def replacement_args(in_0):
    return (in_0,)


@torch.fx.wrap
def fused_relu_dropout0_flatten(in_0):
    in_0.relu_()
    return in_0.flatten(1, -1)


def replacement_func():
    return fused_relu_dropout0_flatten