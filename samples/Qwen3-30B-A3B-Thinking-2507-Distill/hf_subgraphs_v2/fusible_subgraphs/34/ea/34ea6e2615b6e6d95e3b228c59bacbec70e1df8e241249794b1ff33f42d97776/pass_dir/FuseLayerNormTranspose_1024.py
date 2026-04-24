import torch
import triton
import triton.language as tl

from pass_dir.shared_ln_kernel import _dispatch


# ── Pattern: layer_norm with normalized_shape (1024,) ─────────────────────────
# Matches the layer_norm node in the float32/1024 graph.

def pattern(in_0, in_1, tmp_7):
    tmp_8 = torch.nn.functional.layer_norm(tmp_7, (1024,), in_1, in_0, 1e-05)
    return tmp_8


def replacement_args(in_0, in_1, tmp_7):
    return (in_0, in_1, tmp_7, "1024")


def replacement_func():
    return _dispatch