import torch
import triton
import triton.language as tl
from pass_dir.shared_relpos_impl import beit_relpos_dispatch


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    meshgrid = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = meshgrid[0]
    tmp_5 = meshgrid[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_8 = tmp_7[(slice(None, None, None), slice(None, None, None), None)]
    tmp_9 = tmp_7[(slice(None, None, None), None, slice(None, None, None))]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    tmp_13 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_13 += 23
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_13
    tmp_16 = tmp_12[(slice(None, None, None), slice(None, None, None), 1)]
    tmp_16 += 23
    tmp_12[(slice(None, None, None), slice(None, None, None), 1)] = tmp_16
    tmp_19 = tmp_12[(slice(None, None, None), slice(None, None, None), 0)]
    tmp_19 *= 47
    tmp_12[(slice(None, None, None), slice(None, None, None), 0)] = tmp_19
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[(slice(1, None, None), slice(1, None, None))] = tmp_23
    tmp_22[(0, slice(0, None, None))] = 2209
    tmp_22[(slice(0, None, None), 0)] = 2210
    tmp_22[(0, 0)] = 2211
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1):
    return (in_0, in_1, 'n24')


def replacement_func():
    return beit_relpos_dispatch