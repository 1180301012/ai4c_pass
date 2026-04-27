"""
Pass: Fuse (0 + in_1 + in_0) add with 2D spatial mean reduction.
"""
import operator
import torch
import torch.fx as fx
import triton
import triton.language as tl
from pass_dir.fused_kernel_impl import shared_dispatch

def _proxy_iadd(self, other):
    return self.tracer.create_proxy('call_function', operator.iadd, (self, other), {})

if not getattr(fx.Proxy, '_ai4c_iadd_patched', False):
    fx.Proxy.__iadd__ = _proxy_iadd
    fx.Proxy._ai4c_iadd_patched = True


def pattern(in_0, in_1):
    tmp_0 = 0 + in_1
    tmp_0 += in_0
    tmp_2 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_2)


def replacement_args(in_0, in_1):
    return (in_0, in_1, in_0, "2")


def replacement_func():
    return shared_dispatch