"""
Optimization pass for BotNet relative attention with N=16 spatial dimension.

Matched subgraph (5 ops, from tmp_10.reshape onward):

  tmp_11 = tmp_10.reshape(4, 256, 256)   # tmp_10 is contiguous [4,16,16,16,16]
  tmp_12 = in_0 + tmp_11
  tmp_13 = tmp_12.softmax(dim=-1)
  matmul_1 = tmp_13 @ in_4
  tmp_15 = matmul_1.transpose(-1, -2)

Shapes:
  in_0  : [4, 256, 256]        float16/bf16
  tmp_10: [4, 16, 16, 16, 16]  float16/bf16  (contiguous: tmp_9 + in_2)
  in_4  : [4, 256, 128]        float16/bf16
  out   : [4, 128, 256]        float16/bf16
"""
import torch
import triton
import triton.language as tl

from pass_dir.shared_botnet_kernels import botnet_attn_dispatch


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(in_0, tmp_10, in_4):
    tmp_11 = tmp_10.reshape(4, 256, 256)
    tmp_12 = in_0 + tmp_11
    tmp_13 = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15 = matmul_1.transpose(-1, -2)
    return tmp_15


# ---------------------------------------------------------------------------
# Replacement args – include route string for shared dispatcher
# ---------------------------------------------------------------------------
def replacement_args(in_0, tmp_10, in_4):
    return (in_0, tmp_10, in_4, "N16")


# ---------------------------------------------------------------------------
# Replacement function – returns the shared dispatcher (same object as N8)
# ---------------------------------------------------------------------------
def replacement_func():
    return botnet_attn_dispatch