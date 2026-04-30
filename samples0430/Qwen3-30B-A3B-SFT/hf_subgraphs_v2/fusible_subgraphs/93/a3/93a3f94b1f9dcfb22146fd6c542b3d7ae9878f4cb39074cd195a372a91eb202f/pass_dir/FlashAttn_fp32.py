"""
Pass: Fuse Flash-Attention (float32 variant)
Matches: QK^T → softmax(dim=-1, dtype=float32) → dropout(p=0, training=False) → matmul(V) → transpose → reshape
Input  : in_0=[1,16,257,80] fp32, in_1=[1,16,80,257] fp32, in_2=[1,16,257,80] fp32
Output : [1, 257, 1280] fp32
Note: bfloat16/float16 models have extra .to(float32) and .to(dtype) ops after softmax.
      float32 models skip those and go straight from softmax to dropout.
"""

import torch
from pass_dir.flash_attn_triton import flash_attn_kernel_wrapper


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    """
    Full flash attention pattern for float32 model.
    *1.0 might be eliminated; .to(float32) and dropout(p=0)/training=False are both no-ops
    → target: softmax → mm(attn,V) → transpose → contiguous → reshape → contiguous
    """
    matmul   = torch.matmul(in_0, in_1)
    tmp_1    = matmul * 1.0
    tmp_2    = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    matmul_1 = torch.matmul(tmp_2, in_2)
    tmp_6    = matmul_1.transpose(1, 2)
    tmp_7    = tmp_6.contiguous()
    tmp_8    = tmp_7.reshape(1, 257, -1)
    tmp_9    = tmp_8.contiguous()
    return (tmp_9,)


# ── Replacement helpers ───────────────────────────────────────────────────────

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "fp32")


def replacement_func():
    return flash_attn_kernel_wrapper