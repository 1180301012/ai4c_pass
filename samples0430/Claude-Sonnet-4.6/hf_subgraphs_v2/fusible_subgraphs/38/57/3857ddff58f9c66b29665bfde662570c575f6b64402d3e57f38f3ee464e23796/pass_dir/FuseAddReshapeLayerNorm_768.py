import torch
from pass_dir.shared_kernels import fused_layernorm_dispatch


# ---------------------------------------------------------------------------
# Pattern: match layer_norm with normalized_shape=(768,) → single output
# The add+reshape remain in the graph (reshape is a free view).
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    out = torch.nn.functional.layer_norm(in_2, (768,), in_1, in_0, 1e-05)
    return out


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_layernorm_dispatch