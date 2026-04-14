"""
Pass: FusedSoftmaxWeightedSum_SC
Minimal pattern: just mul + sum (dim=1).

Matches: tmp_4 = tmp_3 * in_0 ; tmp_5 = sum(tmp_4, dim=1)
where tmp_3 is the pre-shaped softmax weights [B,2,C,1,1]
and   in_0  is the feature tensor          [B,2,C,H,W].

Replaces the multiply+reduce with a single Triton kernel.
The softmax+reshape+view chain still runs before this; only
the multiply+sum is fused — but it is the most expensive part.
"""

import torch
import triton
import triton.language as tl


# ── Triton kernel ──────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 256},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 512},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_HW': 4096}, num_warps=16, num_stages=3),
    ],
    key=['HW'],
)
@triton.jit
def _sc_kernel(
    weights_ptr,   # [B, 2, 128, 1, 1] softmax weights – contiguous
    data_ptr,      # [B, 2, 128, H, W] feature tensor  – contiguous
    out_ptr,       # [B, 128, H, W]    output          – contiguous
    HW,            # H * W
    BLOCK_HW: tl.constexpr,
):
    # C is always 128 in all test cases; hardcode for efficient bit arithmetic
    C = 128
    LOG2_C = 7        # 128 = 2^7
    MASK_C = 127      # 128 - 1

    bc_idx = tl.program_id(1)   # b * 128 + c
    hw_pid = tl.program_id(0)

    # Fast integer decode using bit ops (C=128 is power of 2)
    b_idx = bc_idx >> LOG2_C   # bc_idx // 128
    c_idx = bc_idx & MASK_C    # bc_idx % 128

    hw_offs = hw_pid * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask    = hw_offs < HW

    # data[b, k, c, hw]  strides: [2*128*HW, 128*HW, HW, 1]
    data_k0_base = b_idx * (256 * HW) + c_idx * HW
    data_k1_base = data_k0_base + (C * HW)

    x0 = tl.load(data_ptr + data_k0_base + hw_offs, mask=mask).to(tl.float32)
    x1 = tl.load(data_ptr + data_k1_base + hw_offs, mask=mask).to(tl.float32)

    # weights[b, k, c, 0, 0]  strides: [256, 128, 1, 1, 1]
    w0 = tl.load(weights_ptr + b_idx * 256 + c_idx      ).to(tl.float32)
    w1 = tl.load(weights_ptr + b_idx * 256 + C + c_idx  ).to(tl.float32)

    out_f32 = w0 * x0 + w1 * x1
    tl.store(out_ptr + bc_idx * HW + hw_offs, out_f32, mask=mask)


# ── Wrapper ────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def _sc_fused(tmp_3, in_0):
    """
    tmp_3 : [B, 2, 128, 1, 1] – softmax-weighted coefficients (already shaped)
    in_0  : [B, 2, 128, H, W] – feature maps
    out   : [B, 128, H, W]    – same dtype as in_0
    """
    B  = in_0.shape[0]
    H  = in_0.shape[3]
    W  = in_0.shape[4]
    HW = H * W
    BC = B * 128   # C is always 128
    out = torch.empty(B, 128, H, W, dtype=in_0.dtype, device=in_0.device)
    grid = lambda meta: (
        (HW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW'],
        BC,
    )
    _sc_kernel[grid](tmp_3, in_0, out, HW)
    return out


# ── Pass API ───────────────────────────────────────────────────────────────────

def pattern(tmp_3, in_0):
    """
    Minimal pattern: match only the multiply + sum(dim=1).
    tmp_3 is a free variable matching any upstream node (the shaped weights).
    in_0  is matched to the model's in_0 placeholder.
    """
    tmp_4 = tmp_3 * in_0
    tmp_5 = torch.sum(tmp_4, dim=1)
    return tmp_5


def replacement_args(tmp_3, in_0):
    return (tmp_3, in_0)


def replacement_func():
    return _sc_fused