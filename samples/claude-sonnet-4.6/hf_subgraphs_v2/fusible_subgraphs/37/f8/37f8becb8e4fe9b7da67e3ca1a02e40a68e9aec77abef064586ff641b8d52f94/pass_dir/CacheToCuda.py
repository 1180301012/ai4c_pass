import torch
from torch import device
from pass_dir.shared_dispatch import _shared_dispatch


# ---------------------------------------------------------------------------
# Pattern / replacement_args / replacement_func  (framework interface)
# ---------------------------------------------------------------------------
def pattern(x):
    return x.to(device(type='cuda'))


def replacement_args(x):
    # Append route tag — _shared_dispatch inspects args[-1] to decide path
    return (x, "cache")


def replacement_func():
    return _shared_dispatch