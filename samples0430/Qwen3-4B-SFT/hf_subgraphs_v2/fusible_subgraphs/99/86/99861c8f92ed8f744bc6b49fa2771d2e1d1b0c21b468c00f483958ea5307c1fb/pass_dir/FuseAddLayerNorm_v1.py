import torch
import triton
import triton.language as tl

from pass_dir.shared import _layernorm_kernel_only


# ── Single-output LN pattern: matches layer_norm only ─────────────────────────
# This matches every graph regardless of return order.
# The preceding `in_2 + in_3` (add residual) stays in the graph untouched.
def pattern(in_0, in_1, in_2):
    out = torch.nn.functional.layer_norm(in_2, (1024,), in_1, in_0, 1e-05)
    return out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return _layernorm_kernel_only