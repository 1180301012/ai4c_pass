"""
Pass 2 of 2: Replace x.mean((2, 3), keepdim=True) with a lookup into
the global mean cache populated by Pass 1 (FusedGeluPrecompMean).

Pattern returns ONE value (the mean output), so _replace_pattern's
length assertion is satisfied.
"""
import torch
from pass_dir.FusedGeluMean_dim23_keepdim import shared_dispatch


# ---------------------------------------------------------------------------
# Pattern: x.mean((2, 3), keepdim=True)  →  single output
# ---------------------------------------------------------------------------

def pattern(x):
    return x.mean((2, 3), keepdim=True)


# replacement_args appends the route string for the cache-read branch.
def replacement_args(x):
    return (x, "mean_cache")


def replacement_func():
    return shared_dispatch