"""
Pass: FuseGeluTransposeAddLayerNorm_p0p1
Matches layer_norm only (single output = tmp_10).
tmp_8 (no-op dropout) stays in graph unchanged; model return stays correct.
Targets float16/float32 graphs — but since pattern is dtype-agnostic, matches all.
"""

import torch
from pass_dir._shared_kernel import _dispatch  # shared replacement func


def pattern(x, w, b):
    return torch.nn.functional.layer_norm(x, (1024,), w, b, 1e-05)


def replacement_args(x, w, b):
    return (x, w, b)


def replacement_func():
    return _dispatch