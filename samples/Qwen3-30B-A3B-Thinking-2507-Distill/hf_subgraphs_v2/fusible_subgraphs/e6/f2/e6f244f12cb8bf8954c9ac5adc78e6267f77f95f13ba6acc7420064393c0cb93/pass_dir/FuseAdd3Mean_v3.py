"""
Pass: FuseAdd3Mean_v3
Matches: 0 + in_1; += in_0  followed by  .mean((2,3), keepdim=True)
Covers: mobileone_s4 start228_end231_3 graphs (two tensor inputs)
"""
import operator
import torch
import torch.fx.proxy as _fx_proxy
from pass_dir.FuseAdd3Mean_v1 import _proxy_iadd
from pass_dir._add_mean_kernel import add_mean_dispatch

_fx_proxy.Proxy.__iadd__ = _proxy_iadd


def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    return (in_1, in_0, in_1, "add2", in_1.dtype)


def replacement_func():
    return add_mean_dispatch