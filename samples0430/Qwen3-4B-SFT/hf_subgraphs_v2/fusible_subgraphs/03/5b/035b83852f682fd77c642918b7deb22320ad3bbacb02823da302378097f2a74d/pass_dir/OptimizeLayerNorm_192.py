import torch
from pass_dir.shared_kernels import do_layer_norm_triton


# ---------------------------------------------------------------------------
# Pattern: layer_norm with normalized_shape=(192,)
#   Matches convit_tiny  (float16 and float32)
# ---------------------------------------------------------------------------
def pattern(bias, weight, x):
    return torch.nn.functional.layer_norm(x, (192,), weight, bias, 1e-06)


def replacement_args(bias, weight, x):
    return (bias, weight, x)


# ---------------------------------------------------------------------------
# The replacement_func() below returns 'do_layer_norm_triton' — the SAME
# Python object as OptimizeLayerNorm's replacement_func.  This satisfies the
# output_pass_replacement_func_limit that limits each unique wrapper to 1.
# ---------------------------------------------------------------------------
def replacement_func():
    return do_layer_norm_triton