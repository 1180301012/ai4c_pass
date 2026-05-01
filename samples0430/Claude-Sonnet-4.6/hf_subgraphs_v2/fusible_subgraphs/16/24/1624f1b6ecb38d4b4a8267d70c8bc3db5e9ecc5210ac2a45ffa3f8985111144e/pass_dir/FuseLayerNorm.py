import torch
from pass_dir.shared_kernels import shared_dispatch_kernel, replacement_func  # noqa: F401


# ---------------------------------------------------------------------------
# Pattern: match a single layer_norm call
#   Returns tmp_13 (single value)
#   tmp_12 (the dropout/preproc output) remains in graph as-is; it feeds
#   both the layer_norm (now replaced) and the model's first return value.
# ---------------------------------------------------------------------------
def pattern(x, ln_weight, ln_bias):
    return torch.nn.functional.layer_norm(x, (768,), ln_weight, ln_bias, 1e-06)


def replacement_args(x, ln_weight, ln_bias):
    # Append "layernorm" route so shared_dispatch_kernel dispatches correctly
    return (x, ln_weight, ln_bias, "layernorm")