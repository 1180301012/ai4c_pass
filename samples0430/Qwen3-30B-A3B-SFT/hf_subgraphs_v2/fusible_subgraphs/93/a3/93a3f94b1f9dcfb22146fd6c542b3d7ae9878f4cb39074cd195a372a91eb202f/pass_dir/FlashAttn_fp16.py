"""
Pass: Fuse Flash-Attention (float16 variant)
Matches: QK^T → *1.0 → softmax → to(float32) → dropout(p=0) → to(float16) → attn×V → transpose → reshape
Input  : in_0=[1,16,257,80] fp16, in_1=[1,16,80,257] fp16, in_2=[1,16,257,80] fp16
Output : [1, 257, 1280] fp16
"""

import torch
from pass_dir.flash_attn_triton import flash_attn_kernel_wrapper


# ── Pattern ──────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    """
    Full flash attention pattern for float16 model.
    QK^T → *1.0 → softmax → to(fp32) → dropout → to(fp16) → mm(attn,V)
    """
    matmul   = torch.matmul(in_0, in_1)
    tmp_1    = matmul * 1.0
    tmp_2    = torch.nn.functional.softmax(tmp_1, dim=-1, dtype=torch.float32)
    tmp_3    = tmp_2.to(torch.float32)
    tmp_4    = torch.nn.functional.dropout(tmp_3, p=0.0, training=False)
    to       = tmp_4.to(torch.float16)
    matmul_1 = torch.matmul(to, in_2)
    tmp_6    = matmul_1.transpose(1, 2)
    tmp_7    = tmp_6.contiguous()
    tmp_8    = tmp_7.reshape(1, 257, -1)
    tmp_9    = tmp_8.contiguous()
    return (tmp_9,)


# ── Replacement helpers ───────────────────────────────────────────────────────

def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2, "fp16")


def replacement_func():
    return flash_attn_kernel_wrapper