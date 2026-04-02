import torch
from torch import device as _device


@torch.fx.wrap
def _fused_to_bool(in_0):
    # Equivalent to in_0.to(bool)[:, arange(N)] which is just in_0.to(bool)
    # since arange(N) = [0,1,...,N-1] selects all columns (identity)
    return in_0.to(dtype=torch.bool)


def pattern(in_0):
    tmp_2 = in_0.to(device=_device(type='cuda', index=0), dtype=torch.bool)
    tmp_3 = torch.arange(2, device=_device(type='cuda', index=0))
    tmp_3 += 0
    tmp_5 = tmp_2[slice(None, None, None), tmp_3]
    return tmp_5


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_to_bool