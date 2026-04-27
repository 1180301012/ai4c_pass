"""
Pass: FuseGeluAddLayerNorm_C128_H16_W12

Fuses gelu(in_2[1,128,16,12]) → flatten → transpose → contiguous
      → add in_3[1,192,128]
      → permute/view/view/permute (no-op)
into a single Triton kernel producing tmp_10 [1,192,128].

The subsequent layer_norm and view remain in the original graph.
Single return value (tmp_10) avoids the multi-output assertion issue.

All three variant passes import the same `fused_dispatch` object so they
pass the output_pass_replacement_func_limit=1 check.
"""

import torch
from pass_dir.shared_dispatch import fused_dispatch


# ── pattern  (takes only in_2 and in_3; returns single tmp_10) ───────────────
def pattern(in_2, in_3):
    tmp_2  = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3  = tmp_2.flatten(2)
    tmp_4  = tmp_3.transpose(1, 2)
    tmp_5  = tmp_4.contiguous()
    tmp_6  = in_3 + tmp_5
    tmp_7  = tmp_6.permute(0, 2, 1)
    tmp_8  = tmp_7.view(1, 128, 16, 12)
    tmp_9  = tmp_8.view(1, 128, -1)
    tmp_10 = tmp_9.permute(0, 2, 1)
    return tmp_10      # ← single return, no tuple


# ── argument extraction ───────────────────────────────────────────────────────
def replacement_args(in_2, in_3):
    return (in_2, in_3, "C128_H16_W12")


# ── replacement entry point  (same object across ALL passes) ─────────────────
def replacement_func():
    return fused_dispatch