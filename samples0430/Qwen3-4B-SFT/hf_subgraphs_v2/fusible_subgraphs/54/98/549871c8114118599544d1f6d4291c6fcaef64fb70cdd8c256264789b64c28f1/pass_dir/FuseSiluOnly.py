import torch
from pass_dir.shared_dispatch import _shared_dispatch

# ---------------------------------------------------------------------------
#  FuseSiluOnly – single-node probe: does torch.nn.functional.silu(x)
#  appear as a node in the compiled graph?
# ---------------------------------------------------------------------------


def pattern(x):
    return torch.nn.functional.silu(x, inplace=False)


def replacement_args(x):
    # probe: pass x directly (dispatch will copy it — wrong semantics,
    # but confirms whether the silu node is present for matching)
    return (x, "silu_probe")


def replacement_func():
    return _shared_dispatch