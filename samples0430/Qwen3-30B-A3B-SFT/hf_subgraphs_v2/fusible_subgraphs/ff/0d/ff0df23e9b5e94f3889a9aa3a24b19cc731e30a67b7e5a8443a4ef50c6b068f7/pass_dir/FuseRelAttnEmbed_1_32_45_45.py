"""
Pass: FuseRelAttnEmbed_1_32_45_45

Matches the fused relative-attention-bias embedding pattern for graphs where:
  - in_0 : (32, 4)   embedding weight on CUDA
  - in_1 : (45, 45)  int64 indices on CPU
  - output: (1, 32, 45, 45)
  - expand shape: (1, -1, 45, 45)

Fuses: to(cuda) + embedding + permute([2,0,1]) + unsqueeze(0) + expand + contiguous
into a single Triton gather kernel that writes directly in the permuted layout.
"""

import torch
import triton
import triton.language as tl
from torch import device

from pass_dir.relative_attn_embed_kernel import _run_fused_embed, _dispatch_fused_embed


# ── Pattern ───────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    tmp_1 = in_1.to(device(type='cuda', index=0))
    tmp_2 = torch.nn.functional.embedding(tmp_1, in_0, None, None, 2.0, False, False)
    tmp_3 = tmp_2.permute([2, 0, 1])
    tmp_4 = tmp_3.unsqueeze(0)
    tmp_5 = tmp_4.expand((1, -1, 45, 45))
    tmp_6 = tmp_5.contiguous()
    return tmp_6


# ── Argument extraction ───────────────────────────────────────────────────────

def replacement_args(in_0, in_1):
    return (in_0, in_1, "1_32_45_45")


# ── Replacement kernel wrapper ────────────────────────────────────────────────

@torch.fx.wrap
def _dispatch_fused_embed(in_0, in_1, route):
    return _run_fused_embed(in_0, in_1, N=45, D_emb=32, D=4, B=1)


def replacement_func():
    return _dispatch_fused_embed