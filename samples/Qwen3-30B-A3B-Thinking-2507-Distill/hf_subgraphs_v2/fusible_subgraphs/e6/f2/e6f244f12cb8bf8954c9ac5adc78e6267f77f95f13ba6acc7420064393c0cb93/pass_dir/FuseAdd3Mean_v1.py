"""
Pass: FuseAdd3Mean_v1
Matches: (in_1 + in_2) + in_0  followed by  .mean((2,3), keepdim=True)
Covers: repvgg_d2se start113_end116_8 graphs
"""
import operator
import torch
import torch.fx.proxy as _fx_proxy
from pass_dir._add_mean_kernel import add_mean_dispatch

# Patch Proxy.__iadd__ so that FX symbolic_trace creates
# call_function(operator.iadd, ...) when tracing the pattern function.
# This makes the pattern match the target graph's iadd nodes.
def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

_fx_proxy.Proxy.__iadd__ = _proxy_iadd


def pattern(in_1, in_2, in_0):
    tmp_0 = in_1 + in_2
    tmp_0 += in_0   # now creates call_function(operator.iadd, ...) via patched __iadd__
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_1, in_2, in_0):
    # a=in_1, b=in_2, c=in_0 → computes in_1+in_2+in_0
    return (in_1, in_2, in_0, "add3", in_1.dtype)


def replacement_func():
    return add_mean_dispatch