"""
Pass: FuseBMMAtOp
Matches:   matmul = in_1 @ in_0
Replaces with a Triton batched-matmul kernel.
The downstream .view() is NOT part of this pattern; it stays in the graph
as a free metadata operation and simply sees the same-shaped output.
"""

import torch
from pass_dir.shared_bmm import triton_batched_matmul


# ---------------------------------------------------------------------------
# Pattern – must mirror model.py exactly (@ operator form)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    matmul = in_1 @ in_0
    return matmul


# ---------------------------------------------------------------------------
# replacement_args – forward the two inputs to the wrapper
# ---------------------------------------------------------------------------

def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# replacement_func – return the callable (do NOT call it)
# ---------------------------------------------------------------------------

def replacement_func():
    return triton_batched_matmul