import torch
from pass_dir.shared_kernels import do_layer_norm_triton


# ---------------------------------------------------------------------------
# Pattern: layer_norm with normalized_shape=(432,)
#   Matches convit_small  (both float16 and float32; bfloat16 in _decomposed)
# ---------------------------------------------------------------------------
def pattern(bias, weight, x):
    return torch.nn.functional.layer_norm(x, (432,), weight, bias, 1e-06)


def replacement_args(bias, weight, x):
    return (bias, weight, x)


# ---------------------------------------------------------------------------
# The replacement_func() below returns 'do_layer_norm_triton' which is the
# SAME Python object as the one returned by OptimizeLayerNorm_192's
# replacement_func — this prevents the framework from dropping passes due to
# the output_pass_replacement_func_limit.
# ---------------------------------------------------------------------------
def replacement_func():
    return do_layer_norm_triton