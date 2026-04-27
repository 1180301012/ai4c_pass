"""
Shared Triton kernel for fused SiLU + AdaptiveAvgPool2d(1) + Flatten.

For each (batch, channel) pair the kernel:
  1. Loads H*W elements from the contiguous NCHW input.
  2. Applies SiLU: x * sigmoid(x)  (computed in fp32 for accuracy).
  3. Accumulates and divides by H*W → global average pool.
Writes the scalar result directly to the [B, C] output (skipping the
trivial flatten and inference-mode dropout).
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 64},  num_warps=2),
        triton.Config({"BLOCK_HW": 64},  num_warps=4),
        triton.Config({"BLOCK_HW": 64},  num_warps=8),
        triton.Config({"BLOCK_HW": 128}, num_warps=2),
        triton.Config({"BLOCK_HW": 128}, num_warps=4),
        triton.Config({"BLOCK_HW": 128}, num_warps=8),
        triton.Config({"BLOCK_HW": 256}, num_warps=4),
        triton.Config({"BLOCK_HW": 256}, num_warps=8),
        triton.Config({"BLOCK_HW": 512}, num_warps=4),
        triton.Config({"BLOCK_HW": 512}, num_warps=8),
    ],
    key=["HW"],
)
@triton.jit
def _silu_avgpool_kernel(
    in_ptr,   # pointer to input  [B, C, H, W] contiguous
    out_ptr,  # pointer to output [B, C]        contiguous
    HW,       # H * W  (runtime int)
    BLOCK_HW: tl.constexpr,
):
    """Each program handles exactly one (b, c) pair."""
    bc = tl.program_id(0)
    base = bc * HW

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for start in range(0, HW, BLOCK_HW):
        offs = start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        # Load; masked positions get 0.0, and silu(0)=0 so they don't affect sum
        x = tl.load(in_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += x * tl.sigmoid(x)

    total = tl.sum(acc, axis=0) / HW
    # tl.store casts float32 → output element dtype automatically
    tl.store(out_ptr + bc, total)


@torch.fx.wrap
def silu_avgpool_fused(x):
    """
    Fused SiLU + global average pool + flatten.

    x  : [B, C, H, W]  (any of fp32 / fp16 / bf16, contiguous NCHW)
    out: [B, C]         same dtype as x

    Returns a tuple (out,) to match the model's `return (tmp_3,)` signature.
    """
    B = x.shape[0]
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3]
    BC = B * C

    out = torch.empty((B, C), dtype=x.dtype, device=x.device)

    _silu_avgpool_kernel[(BC,)](
        x, out,
        HW=HW,
    )

    return (out,)