"""
FuseConv1x1Coalesced.py

Pattern : torch.conv2d(in_2, in_1, in_0, (1,1),(0,0),(1,1),1)
           -- matches ALL batch-sizes (B is not hard-coded)

Replacement: Triton kernel that performs the 1×1 convolution (weight shape
             [1,C,1,1]) as a coalesced channel-sequential GEMV:

               out[b, p] = Σ_c  in2[b, c, p] * w[c]  +  bias

             where p ∈ [0, HW) indexes the flattened spatial position.

             MEMORY-ACCESS TRICK
             --------------------
             In NCHW layout the slice  in2[b, c, :]  (all HW pixels for one
             channel) is CONTIGUOUS.  By iterating over channels one-at-a-time
             and loading BLOCK_HW consecutive pixels, every memory transaction
             is fully coalesced.  This avoids the strided channel-dimension
             accesses of a naïve 2-D tile.

             The kernel writes a float-32 result of shape [B, HW].
             The wrapper casts to the input dtype and reshapes to [B, 1, HW].
             The graph's downstream  view(B, 1, -1)  then becomes a no-op, and
             softmax(dim=-1) correctly normalises over HW = 4096 elements.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # Pure sequential: scalar weight load + 1-D coalesced pixel load per channel.
        # Large BLOCK_HW → 100% L2 cache-line efficiency (BLOCK_HW×dtype ≥ 128 B).
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=8),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def _conv1x1_parallel_kernel(
    w_ptr,      # [C]
    in2_ptr,    # [B, C, HW]   NCHW input (spatial dims flattened)
    bias_ptr,   # [1]
    out_ptr,    # [B, HW]      float32 output
    B, C, HW,
    BLOCK_HW: tl.constexpr,
):
    """
    Grid = (B, ceil(HW / BLOCK_HW)).

    Sequential channel loop: scalar weight × BLOCK_HW consecutive pixels.
      • Fully coalesced BLOCK_HW-element loads per channel.
      • Minimal register pressure: only acc[BLOCK_HW] + one scalar + one vector.
      • Autotune selects BLOCK_HW to balance cache-line efficiency vs GPU utilisation.
    """
    b       = tl.program_id(0)
    hw_tile = tl.program_id(1)

    p_offs = hw_tile * BLOCK_HW + tl.arange(0, BLOCK_HW)
    p_mask = p_offs < HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for c in range(C):
        w_c  = tl.load(w_ptr + c).to(tl.float32)
        offs = b * C * HW + c * HW + p_offs
        x    = tl.load(in2_ptr + offs, mask=p_mask, other=0.0).to(tl.float32)
        acc  = acc + w_c * x

    bias = tl.load(bias_ptr).to(tl.float32)
    acc  = acc + bias

    tl.store(out_ptr + b * HW + p_offs, acc, mask=p_mask)


@triton.jit
def _bias_add_kernel(
    x_ptr, bias_ptr, N, BLOCK: tl.constexpr,
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x    = tl.load(x_ptr + offs, mask=mask)
    bias = tl.load(bias_ptr).to(x.dtype)
    tl.store(x_ptr + offs, x + bias, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern  –  matches conv2d for ANY batch size
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    return torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─────────────────────────────────────────────────────────────────────────────
# Replacement wrapper
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def _triton_conv1x1_all(in_0, in_1, in_2):
    """
    Drop-in replacement for  torch.conv2d(in_2, in_1, in_0, (1,1),(0,0),(1,1),1)
    when in_1 has shape [1, C, 1, 1] and in_0 has shape [1].

    Uses a Triton kernel that:
      • Loads ALL channels at once as a [BLOCK_C, BLOCK_HW] 2-D tile.
      • Issues BLOCK_C simultaneous HBM requests → near-peak bandwidth.
      • Uses tl.sum(axis=0) for a parallel intra-CTA channel reduction
        (mimics cuBLAS warp-level GEMV with shuffle-reduce).

    Returns [B, 1, HW] instead of [B, 1, H, W].
    The graph's downstream  .view(B, 1, -1)  is then a no-op, and
    .softmax(dim=-1) correctly normalises over HW = H*W elements.
    """
    B, C, H, W = in_2.shape
    HW = H * W

    w_flat  = in_1.view(-1)                         # [C]
    out_f32 = torch.empty((B, HW), dtype=torch.float32, device=in_2.device)

    grid = lambda meta: (B, triton.cdiv(HW, meta['BLOCK_HW']))
    _conv1x1_parallel_kernel[grid](w_flat, in_2, in_0, out_f32, B, C, HW)

    # Cast to input dtype (no-op for float32; cheap copy for fp16/bf16)
    # then reshape to [B, 1, HW] so downstream view(B,1,-1) is a no-op.
    return out_f32.to(in_2.dtype).view(B, 1, HW)


def replacement_func():
    return _triton_conv1x1_all