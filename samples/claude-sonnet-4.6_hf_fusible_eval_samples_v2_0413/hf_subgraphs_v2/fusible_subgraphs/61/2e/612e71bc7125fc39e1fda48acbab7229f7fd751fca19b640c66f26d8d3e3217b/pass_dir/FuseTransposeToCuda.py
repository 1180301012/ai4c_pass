"""
Fuses:
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))

in_0 is [1, N] bfloat16 already on CUDA.  Both [1, N] and [N, 1] share
the same linear memory layout, so we materialise the output shape with a
plain Triton copy kernel.

Imports the shared _dispatch object from pass_dir/_shared.py so that
replacement_func() returns the *exact same* Python object as
FuseL2Normalize.py, satisfying the framework's replacement_func_limit.
"""

import torch
from torch import device
from pass_dir._shared import _dispatch  # noqa: F401  (imported for re-export)


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in_0):
    tmp_2 = in_0.t()
    tmp_3 = tmp_2.to(device(type='cuda'))
    return tmp_3


def replacement_args(in_0):
    return (in_0, "transpose_to_cuda")


def replacement_func():
    return _dispatch