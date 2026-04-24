"""
Pass: FuseAdd3Mean_v3b
Matches: 0 + in_0; += 0  followed by  .mean((2,3), keepdim=True)
Covers: mobileone_s4 start356_end359_12 graphs (one tensor input, two scalar zeros)
"""
import operator
import torch
import torch.fx.proxy as _fx_proxy
from pass_dir.FuseAdd3Mean_v1 import _proxy_iadd
from pass_dir._add_mean_kernel import add_mean_dispatch

_fx_proxy.Proxy.__iadd__ = _proxy_iadd


def pattern(in_0):
    tmp_0 = 0 + in_0
    tmp_0 += 0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0):
    return (in_0, in_0, in_0, "add0", in_0.dtype)


def replacement_func():
    return add_mean_dispatch