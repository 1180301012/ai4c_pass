"""
Pass: FuseMatmulPermute
Matches: torch.matmul(x, y).permute(0,2,1,3).contiguous()
Covers:  ALL 6 model variants (bfloat16 + float16, all 3 ConvBert sizes)

Avoids the full-attention pattern matching issues with dropout tracing.
The pattern starts AFTER softmax/dropout, taking the attention-weight tensor
as a placeholder argument, so dropout never appears in the pattern graph.
"""
import torch
import triton
import triton.language as tl
from pass_dir.attn_kernel import matmul_permute_kernel, fused_matmul_permute


def pattern(x, y):
    m  = torch.matmul(x, y)
    p  = m.permute(0, 2, 1, 3)
    return p.contiguous()


def replacement_args(x, y):
    return (x, y)


def replacement_func():
    return fused_matmul_permute