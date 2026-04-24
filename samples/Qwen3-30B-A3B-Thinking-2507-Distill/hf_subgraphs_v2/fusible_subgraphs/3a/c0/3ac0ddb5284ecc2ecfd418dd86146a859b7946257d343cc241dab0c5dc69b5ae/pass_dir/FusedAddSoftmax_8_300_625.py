import torch
from pass_dir.shared_add_softmax import fused_add_softmax_dispatch


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    # Match add + softmax only; views/dropout remain as no-ops in the graph
    tmp_0 = in_1 + in_0
    tmp_2 = torch.nn.functional.softmax(tmp_0, dim=-1)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    # Returns the SAME function object as FusedAddSoftmax_8_625_625
    return fused_add_softmax_dispatch