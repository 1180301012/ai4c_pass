import torch
from pass_dir.shared_subgraph_impl import _dispatch_full_subgraph


# ── Pattern: broadcast multiply only (single-tensor return — avoids tuple crash)
# expand_as and new_zeros are NOT included here — they remain in the compiled
# graph as cheap view / CUDA-memset operations.

def pattern(in_1, in_2):
    tmp_0 = in_1.view(-1, 1)
    tmp_1 = tmp_0 * in_2
    return tmp_1


def replacement_args(in_1, in_2):
    # Use a unique route string per pass so the dispatch wrapper can identify
    # which variant (same function object → satisfies replacement_func_limit).
    return (in_1, in_2, "1000_16")


def replacement_func():
    return _dispatch_full_subgraph