import torch
import triton
import triton.language as tl
from pass_dir.shared_dispatch import _fused_replacement


# ---------------------------------------------------------------------------
# Fused GELU + spatial-mean pass.
#
# Pattern: matches the full gelu → mean(dims 2,3, keepdim) chain.
#          Both outputs are returned because both appear in the model's return.
#
# Replacement: _fused_replacement(in_0)
#   - NOT @torch.fx.wrap'd, so the FX tracer enters it and sees:
#       (a) one opaque call_function node for _fused_gelu_mean (wrapped)
#       (b) two operator.getitem nodes for result[0] and result[1]
#   - The two getitem outputs map to the pattern's (tmp_0, tmp_1) outputs.
#   - Inside _fused_gelu_mean the single Triton kernel reads input ONCE and
#     writes both GELU output and spatial-mean — saving one full tensor read
#     vs running GELU and mean as separate kernels.
# ---------------------------------------------------------------------------
def pattern(in_0):
    tmp_0 = torch.nn.functional.gelu(in_0)
    tmp_1 = tmp_0.mean((2, 3), keepdim=True)
    return (tmp_0, tmp_1)


def replacement_args(in_0):
    return (in_0,)


def replacement_func():
    return _fused_replacement