"""
Pass: FuseAdd3Mean_v2
Matches: (in_0 + in_1) + in_2  followed by  .mean((2,3), keepdim=True)
Covers: repvgg_d2se start181_end184_14 graphs
"""
import operator
import torch
import torch.fx.proxy as _fx_proxy
from pass_dir.FuseAdd3Mean_v1 import _proxy_iadd
from pass_dir._add_mean_kernel import add_mean_dispatch

# Apply the same patch so FX creates call_function(operator.iadd,...) in pattern traces.
_fx_proxy.Proxy.__iadd__ = _proxy_iadd


def pattern(in_0, in_1, in_2):
    tmp_0 = in_0 + in_1
    tmp_0 += in_2
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1, in_2):
    # a=in_0, b=in_1, c=in_2 → computes in_0+in_1+in_2
    return (in_0, in_1, in_2, "add3", in_0.dtype)


def replacement_func():
    return add_mean_dispatch