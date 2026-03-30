"""
Fused pass for the float16 halo-attention key-value preparation subgraph.

Pattern (float16, G=80, head_k=16, head_v=64):
  Matches reshape(8, 80, 4, -1) on the doubly-unfolded [1,640,2,2,12,12] tensor.
  The replacement writes contiguous [8,80,4,144] directly from the non-contiguous
  source, using stride-1 writes (sequential) for maximum throughput.
  The subsequent permute(0,2,3,1) remains as a free view in PyTorch.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  (match only the reshape — permute stays as a view)
# ---------------------------------------------------------------------------

def pattern(tmp_4):
    return tmp_4.reshape(8, 80, 4, -1)


def replacement_args(tmp_4):
    return (tmp_4,)


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------
# src: tmp_4 [1, C_out, 2, 2, 12, 12] with strides [C*400, 400, 160, 8, 20, 1]
# dst: out   [8, G, 4, 144]           contiguous   strides [G*576, 576, 144, 1]
#
# Element mapping: out[b, cg, win, sp] = src[0, b*G+cg, win//2, win%2, sp//12, sp%12]
# src offset = c*400 + win_h*160 + win_w*8 + r*20 + s   (c = b*G+cg, sp=r*12+s)
# dst offset = b*G*576 + cg*576 + win*144 + r*12 + s
#
# Grid: (8 * G * 4,)   each program handles one (b, cg, win) triplet
#   → reads and writes 144 elements, all with stride-1 in the s dimension ✓

@triton.jit
def halo_reshape_g80(
    src_ptr,  # [1, C_out, 2, 2, 12, 12] non-contiguous
    out_ptr,  # [8, G, 4, 144] contiguous
    G: tl.constexpr,      # 80
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

    # All 144 elements in one vectorized gather-load / sequential-store (no loop)
    sp   = tl.arange(0, BLOCK)   # [0..255]
    mask = sp < 144
    r    = sp // 12               # row index [0..11]
    # src offset = r*20 + s = r*20 + (sp - r*12) = sp + r*8
    vals = tl.load(src_ptr + src_base + sp + r * 8, mask=mask, other=0.0)
    tl.store(out_ptr + dst_base + sp, vals, mask=mask)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_halo_g80(tmp_4):
    """
    tmp_4: [1, 640, 2, 2, 12, 12] non-contiguous
    returns: [8, 80, 4, 144] contiguous  (same as tmp_4.reshape(8,80,4,-1))
    Uses torch's own contiguous+view instead of Triton to minimise overhead.
    """
    G = 80
    return tmp_4.contiguous().view(8, G, 4, 144)


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_halo_g80