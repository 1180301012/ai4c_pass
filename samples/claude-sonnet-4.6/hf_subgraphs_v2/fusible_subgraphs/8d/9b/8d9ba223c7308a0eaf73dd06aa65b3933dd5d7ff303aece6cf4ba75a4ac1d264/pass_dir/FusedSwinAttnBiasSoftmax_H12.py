"""
Optimization pass for Swin Transformer attention bias + softmax (12 heads).

Matches the subgraph starting after linear() through dropout and replaces it
with a single fused Triton kernel that performs:
  gather → sigmoid×16 → add(attn_scores) → add(2×attn_mask) → softmax

Covers both bfloat16 and float16 variants of the 12-head configuration:
  - in_2 shape: [64, 12, 64, 64]
  - in_3 shape: [64, 64, 64]
"""

import torch
from pass_dir.swin_attn_kernel import fused_swin_attn


def pattern(linear_out, in_0, in_2, in_3):
    """
    Matches the post-linear subgraph for 12 attention heads.

    linear_out: output of torch.nn.functional.linear  [1, 15, 15, 12]
    in_0:       relative position index buffer        [64, 64]  int64
    in_2:       attention scores                      [64, 12, 64, 64]
    in_3:       attention mask                        [64, 64, 64]
    """
    tmp_3 = linear_out.view(-1, 12)
    tmp_4 = in_0.view(-1)
    tmp_5 = tmp_3[tmp_4]
    tmp_6 = tmp_5.view(64, 64, -1)
    tmp_7 = tmp_6.permute(2, 0, 1)
    tmp_8 = tmp_7.contiguous()
    tmp_9 = torch.sigmoid(tmp_8)
    tmp_10 = 16 * tmp_9
    tmp_11 = tmp_10.unsqueeze(0)
    tmp_12 = in_2 + tmp_11
    tmp_13 = tmp_12.view(1, 64, 12, 64, 64)
    tmp_14 = in_3.unsqueeze(1)
    tmp_15 = tmp_14.unsqueeze(0)
    tmp_16 = tmp_13 + tmp_15
    tmp_17 = in_3.unsqueeze(1)
    tmp_18 = tmp_17.unsqueeze(0)
    tmp_19 = tmp_16 + tmp_18
    tmp_20 = tmp_19.view(-1, 12, 64, 64)
    tmp_21 = torch.nn.functional.softmax(tmp_20, dim=-1)
    tmp_22 = torch.nn.functional.dropout(tmp_21, 0.0, False, False)
    return (tmp_22,)


def replacement_args(linear_out, in_0, in_2, in_3):
    return (linear_out, in_0, in_2, in_3)


def replacement_func():
    return fused_swin_attn