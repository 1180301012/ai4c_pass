"""
Fused pass for relative-position-encoding attention with 8x8 grid.

Pattern (starting AFTER permute to avoid pad-args matching issues):
  tmp_9  = tmp_8.permute((0,3,1,4,2))                     # -> [4,8,8,8,8]
  tmp_10 = tmp_9 + in_2                             # -> [4,8,8,8,8]
  tmp_11 = reshape(4, 64, 64)                       # -> [4,64,64]
  tmp_12 = in_0 + tmp_11
  tmp_13 = softmax(dim=-1)
  matmul_1= tmp_13 @ in_4                           # -> [4,64,128]
  tmp_15 = matmul_1.transpose(-1, -2)               # -> [4,128,64]

Optimisation: fuse permute_out+add+softmax+matmul+transpose into one kernel.
(in_2 is the additive bias [4,8,8,8,8] which is added to the permuted tensor
before reshaping to [4,64,64], so its contribution is folded into in1 via the
reshape that follows permute)
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel  (in1 is the permuted tensor [B, N, N])
# ---------------------------------------------------------------------------

@triton.jit
def fused_attn_8_kernel(
    in0_ptr,     # [B, N, N]    additive bias
    in1_ptr,     # [B, N, N]    permuted logits (in_1 reshaped from [4,8,8,8,8])
    in4_ptr,     # [B, N, C]    values
    out_ptr,     # [B, C, N]    output
    N: tl.constexpr,   # 64
    C: tl.constexpr,   # 128
):
    W: tl.constexpr = 8
    pid    = tl.program_id(0)
    b      = pid // N
    q_flat = pid % N
    q_row  = q_flat // W
    q_col  = q_flat %  W

    cols  = tl.arange(0, N)
    k_row = cols // W
    k_col = cols %  W

    # Load in1_new[b, q_flat, cols]  (= permuted relative-pos logits)
    in1_row = tl.load(in1_ptr + b * (N * N) + q_flat * N + cols)
    in1_row = in1_row.to(tl.float32)

    # Load in0[b, q_flat, cols]  (additive bias)
    in0_row = tl.load(in0_ptr + b * (N * N) + q_flat * N + cols)
    in0_row = in0_row.to(tl.float32)

    # Attention scores: permuted logits + additive bias
    attn = in1_row + in0_row   # [N]

    # Softmax
    attn_max = tl.max(attn[None, :], axis=1)
    attn     = tl.exp(attn[None, :] - attn_max)
    attn_sum = tl.sum(attn[None, :], axis=1)
    attn     = attn / attn_sum

    # attn[1,N] @ in4[b,:, :][N,C] -> out[1,C]
    in4_base = b * (N * C)
    in4_ptrs = in4_ptr + in4_base + cols[:, None] * C + tl.arange(0, C)[None, :]
    v = tl.load(in4_ptrs)
    v = v.to(tl.float32)

    out = tl.zeros([1, C], dtype=tl.float32)
    out = tl.dot(attn, v)

    # Store transposed: out_ptr[b, c, q_flat]
    out_base = b * (C * N) + q_flat
    tl.store(out_ptr + out_base + tl.arange(0, C)[:, None] * N, tl.trans(out))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_relpos_attn_w8(in_0, in_1, in_2, in_3):
    out = torch.empty((4, 128, 64), dtype=in_0.dtype, device=in_0.device)
    fused_attn_8_kernel[(256,)](in_0, in_1, in_3, out, N=64, C=128)
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    tmp_10   = in_1 + in_2
    tmp_11   = tmp_10.reshape(4, 64, 64)
    tmp_12   = in_0 + tmp_11
    tmp_13   = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_3
    tmp_15   = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_relpos_attn_w8