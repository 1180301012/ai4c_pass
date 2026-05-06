"""
Pass: FuseLayerNormDropoutViewPermute_96

Matches:  layer_norm(x, (96,), weight, bias, 1e-05)
          dropout(result, 0.0, False, False)

Returns:  tmp_9 (the identity dropout output = layer_norm output)

The downstream view/pad/view/permute are cheap metadata ops; fusing
layer_norm + dropout eliminates one kernel call and produces a Triton-based
normalization instead of PyTorch's reference implementation.
"""
import torch
import triton
import triton.language as tl
from pass_dir.shared_layer_norm_kernel import fused_dispatch, replacement_func  # noqa: F401


# ---------------------------------------------------------------------------
# Pattern  (stop BEFORE the no-op pad which causes TARGET_MISMATCH in FX)
# ---------------------------------------------------------------------------
def pattern(x, weight, bias):
    ln_out = torch.nn.functional.layer_norm(x, (96,), weight, bias, 1e-05)
    dropped = torch.nn.functional.dropout(ln_out, 0.0, False, False)
    return dropped


# ---------------------------------------------------------------------------
# Replacement args
# ---------------------------------------------------------------------------
def replacement_args(x, weight, bias):
    return (x, weight, bias, "route_96")


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_dispatch