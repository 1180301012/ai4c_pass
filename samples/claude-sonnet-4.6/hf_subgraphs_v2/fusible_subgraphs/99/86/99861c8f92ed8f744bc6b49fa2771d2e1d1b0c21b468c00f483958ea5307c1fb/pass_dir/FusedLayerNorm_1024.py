"""
Pass: FusedLayerNorm_1024

Matches layer_norm with normalized_shape=(1024,) and replaces it with a fast
Triton kernel.  The add that feeds into layer_norm is left untouched as a
placeholder, so the pattern returns only ONE value (the layer_norm output),
avoiding the multi-output returning_nodes assertion failure.

Works for ALL four target graphs regardless of return-tuple ordering:
  - return (tmp_2, tmp_4)   [Aniemore / unispeech-sat]
  - return (tmp_4, tmp_2)   [galsenai / hubert-large]

Design notes
------------
* @torch._dynamo.disable on pattern()  – prevents recursive dynamo re-tracing
  that occurs when torch.nn.functional.layer_norm is called with FX proxies
  while the torch.compile backend is active.
* _triton_ln is imported from the CACHED module pass_dir.layernorm_kernel so
  the same Python object is returned by replacement_func() on every module
  reload, satisfying the set_g_replacement_func() identity assertion.
"""
import torch
import torch._dynamo
from pass_dir.layernorm_kernel import _triton_ln   # cached module → stable identity


# ---------------------------------------------------------------------------
# Pattern: match layer_norm(in_2, (1024,), weight, bias, eps) -> 1 output
# in_2 is a wildcard placeholder that maps to tmp_2 (the add result)
# @torch._dynamo.disable prevents recursive dynamo tracing of layer_norm
# ---------------------------------------------------------------------------
@torch._dynamo.disable
def pattern(in_0, in_1, in_2):
    return torch.nn.functional.layer_norm(in_2, (1024,), in_1, in_0, 1e-05)


# ---------------------------------------------------------------------------
# Argument extractor — computes output buffer + num_rows as regular FX graph
# nodes (on real tensors), keeping them outside with_dispatch_wrapper_run
# ---------------------------------------------------------------------------
def replacement_args(in_0, in_1, in_2):
    out      = torch.empty_like(in_2)        # allocated as regular FX node
    num_rows = in_2.numel() // 1024          # computed as regular FX node
    return (in_0, in_1, in_2, out, num_rows)


# ---------------------------------------------------------------------------
# Replacement factory – returns the SAME cached function on every call
# ---------------------------------------------------------------------------
def replacement_func():
    return _triton_ln