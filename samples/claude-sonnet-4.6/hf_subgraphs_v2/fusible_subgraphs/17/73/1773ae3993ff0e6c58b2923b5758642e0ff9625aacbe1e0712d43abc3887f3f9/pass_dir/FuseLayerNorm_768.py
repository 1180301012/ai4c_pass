"""
Optimization pass: fuse layer_norm with hidden_size=768 via a Triton kernel.
Matches graphs: hustvl_yolos-base (float16).

Uses the shared replacement_func routing technique: all FuseLayerNorm_* pass
files import and return the SAME _dispatch_layer_norm object so the framework
does not drop passes due to output_pass_replacement_func_limit.
"""
import torch

from pass_dir.triton_layer_norm_kernel import _dispatch_layer_norm


# ---------------------------------------------------------------------------
# Pattern: must mirror model.py exactly
#   tmp_3 = torch.nn.functional.layer_norm(in_4, (768,), in_1, in_0, 1e-12)
# in_0 = bias, in_1 = weight, in_4 = input activation
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_4):
    return torch.nn.functional.layer_norm(in_4, (768,), in_1, in_0, 1e-12)


def replacement_args(in_0, in_1, in_4):
    # Append route string so the shared dispatcher picks the right kernel.
    return (in_0, in_1, in_4, "route_768")


def replacement_func():
    return _dispatch_layer_norm