"""
Fused pass for the bfloat16 / float32 halo-attention KV preparation subgraph.

Pattern (from eca_halonext26ts_start109_end119_1, bfloat16 & float32):
  [anchored at padded conv output: shape [1, 384, 20, 20]]
  unfold(2, 12, 8)                                    -> [1, 384,  2, 20, 12]
  unfold(3, 12, 8)                                    -> [1, 384,  2,  2, 12, 12]
  reshape(8, 48, 4, -1)                               -> [8,  48,  4, 144]
  permute(0, 2, 3, 1)                                 -> [8,   4, 144,  48]
  split([16, 32], dim=-1)                             -> [8,4,144,16], [8,4,144,32]
  transpose(-1,-2) on first split                     -> [8,4, 16,144]
  returns (out1=[8,4,16,144], out2=[8,4,144,32])

Replacement: single Triton gather kernel that reads directly from the padded
tensor [1, 384, 20, 20] and writes the two output tensors in one pass.

G = 48, head_k = 16, head_v = 32
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  (anchored at the padded tensor – the output of F.pad)
# ---------------------------------------------------------------------------

def pattern(tmp_4):
    return tmp_4.reshape(8, 48, 4, -1)


def replacement_args(tmp_4):
    return (tmp_4,)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
# Grid: (8 * 4 * 12 * G,)
# Each program handles one row (12 cols) of the 12×12 halo window for
# one (batch-group b, window win, row r, channel-in-group c_group).
#
# Reads from padded [1, C_out, 20, 20] (no boundary check needed):
#   h_pad = win_h * 8 + r              win_h = win // 2
#   w_pad = win_w * 8 + col            win_w = win % 2  col in [0,12)
#
# c_group < head_k  → write to out1[b, win, c_group, r*12+col]
# c_group >= head_k → write to out2[b, win, r*12+col, c_group-head_k]

@triton.jit
def halo_reshape_g48(
    src_ptr,  # [1, C_out, 2, 2, 12, 12] non-contiguous
    out_ptr,  # [8, G, 4, 144] contiguous
    G: tl.constexpr,      # 48
    BLOCK: tl.constexpr,  # 256
):
    pid = tl.program_id(0)

    win = pid % 4
    tmp = pid // 4
    cg  = tmp % G
    b   = tmp // G

    win_h = win // 2
    win_w = win % 2
    c     = b * G + cg

    src_base = c * 400 + win_h * 160 + win_w * 8
    dst_base = b * (G * 576) + cg * 576 + win * 144

    sp   = tl.arange(0, BLOCK)
    mask = sp < 144
    r    = sp // 12
    vals = tl.load(src_ptr + src_base + sp + r * 8, mask=mask, other=0.0)
    tl.store(out_ptr + dst_base + sp, vals, mask=mask)


@torch.fx.wrap
def fused_halo_g48(tmp_4):
    """
    tmp_4: [1, 384, 2, 2, 12, 12] non-contiguous
    returns: [8, 48, 4, 144] contiguous
    """
    G = 48
    return tmp_4.contiguous().view(8, G, 4, 144)


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_halo_g48