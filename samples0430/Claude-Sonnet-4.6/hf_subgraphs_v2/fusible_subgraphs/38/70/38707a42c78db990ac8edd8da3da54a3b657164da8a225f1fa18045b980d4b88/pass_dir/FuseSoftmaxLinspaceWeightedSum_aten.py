import torch
import triton
import triton.language as tl

# Import the shared dispatch from pass 1 (loaded first, registered in sys.modules)
from FuseSoftmaxLinspaceWeightedSum import shared_kernel_dispatch


# ---------------------------------------------------------------------------
# Pattern: PARTIAL FUSION (fallback if full-fusion pattern does not match)
#   softmax_out  = whatever node computes F.softmax(in_0, dim=1)
#   linspace_tensor = the linspace(0,4,5) node
#   Matches the Python-level ops dynamo records for (softmax_out * linspace)
#   without needing to know the exact softmax node representation.
# ---------------------------------------------------------------------------

def pattern(softmax_out, linspace_tensor):
    tmp_2 = softmax_out * linspace_tensor
    tmp_3 = tmp_2.sum(dim = 1)
    tmp_4 = 5 - tmp_3
    return tmp_4


def replacement_args(softmax_out, linspace_tensor):
    return (softmax_out, "partial_fusion")


def replacement_func():
    return shared_kernel_dispatch