"""
Full-model fusion pass for H=1024 (float32 and bfloat16 models).

Pattern mirrors model.py EXACTLY:
  arange → unsqueeze → +=2 → view → index_select → view → detach → to
  → add → dropout(training=False) → layer_norm((1024,))
  → returns (tmp_13, tmp_14)

Replacement: single Triton kernel that loads in_0 + in_1[row+2],
             computes add+LN, returns (sum_out, ln_out).

Falls back gracefully: if this pattern fails to match, the fallback
FuseAddDropoutLayerNorm_H1024 pass (layer-norm only) will still apply.
"""
import torch
import triton          # noqa: F401
import triton.language as tl  # noqa: F401
from torch import device

from pass_dir.shared_dispatch import shared_fused_add_ln


# ---------------------------------------------------------------------------
# Pattern: exact mirror of model.py forward (H=1024)
# IMPORTANT: no tmp_x = None cleanup; no alias assignments.
# The tuple return (tmp_13, tmp_14) covers both observable outputs.
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_4 = torch.arange(0, 9, dtype=torch.int64, device=device(type='cuda', index=0))
    tmp_5 = tmp_4.unsqueeze(0)
    tmp_5 += 2
    tmp_7 = tmp_5.view(-1)
    tmp_8 = in_1.index_select(0, tmp_7)
    tmp_9 = tmp_8.view(1, 9, 1024)
    tmp_10 = tmp_9.detach()
    tmp_11 = tmp_10.to(device(type='cuda', index=0))
    tmp_12 = in_0 + tmp_11
    tmp_13 = torch.nn.functional.dropout(tmp_12, p=0.1, training=False)
    tmp_14 = torch.nn.functional.layer_norm(tmp_13, (1024,), in_3, in_2, 1e-05)
    return (tmp_13, tmp_14)


def replacement_args(in_0, in_1, in_2, in_3):
    # a=in_0, b=in_1, c=in_2(bias), d=in_3(weight)
    return (in_0, in_1, in_2, in_3, "full_1024")


def replacement_func():
    return shared_fused_add_ln