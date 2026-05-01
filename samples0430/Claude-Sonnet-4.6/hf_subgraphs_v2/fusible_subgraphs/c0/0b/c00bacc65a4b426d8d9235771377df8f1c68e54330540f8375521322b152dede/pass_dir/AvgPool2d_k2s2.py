import torch
from pass_dir.kernels import _do_avgpool


def pattern(x):
    return torch.nn.functional.avg_pool2d(x, 2, 2, 0, True, False, None)


def replacement_args(x):
    return (x,)


@torch.fx.wrap
def triton_avg_pool2d_k2s2(x):
    """Direct wrapper — no routing overhead, single @torch.fx.wrap call."""
    return _do_avgpool(x)


def replacement_func():
    return triton_avg_pool2d_k2s2