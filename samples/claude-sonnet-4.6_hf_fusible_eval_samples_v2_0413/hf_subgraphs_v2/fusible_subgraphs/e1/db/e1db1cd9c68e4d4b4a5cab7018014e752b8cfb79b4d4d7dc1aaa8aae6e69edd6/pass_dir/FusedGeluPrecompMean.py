"""
Pass 1 of 2: Replace gelu(x) with a Triton kernel that computes both
GELU and the spatial mean in a single pass.  The mean is stored in a
global cache (_MEAN_CACHE) as a side-effect; Pass 2 reads it back.

Pattern returns ONE value (the GELU output), so _replace_pattern's
length assertion is satisfied.
"""
import torch
from pass_dir.FusedGeluMean_dim23_keepdim import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern: gelu(x)  →  single output
# ---------------------------------------------------------------------------

def pattern(x):
    return torch.nn.functional.gelu(x)


# replacement_args appends the route string so the shared dispatch can
# tell which branch to execute at runtime.
def replacement_args(x):
    return (x, "gelu_mean")


def replacement_func():
    return shared_dispatch