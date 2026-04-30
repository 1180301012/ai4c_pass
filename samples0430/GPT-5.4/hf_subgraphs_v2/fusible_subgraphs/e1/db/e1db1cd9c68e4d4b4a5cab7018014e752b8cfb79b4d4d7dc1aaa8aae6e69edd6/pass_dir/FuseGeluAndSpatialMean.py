import torch
import triton
import triton.language as tl
from pass_dir.shared_fused_kernels import shared_replacement_func


def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


def replacement_args(in_0):
    return (in_0, 'fused')


def replacement_graph(in_0, route):
    outs = shared_replacement_func()(in_0, route)
    return (outs[0], outs[1])


def replacement_func():
    return replacement_graph