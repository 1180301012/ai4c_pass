"""
Hybrid: best-of-both-worlds for fused 1x1 conv + softmax.

The single-kernel (1 CTA per batch) is currently our best performer because
the inline softmax avoids a temp buffer + second kernel launch.
Use high num_warps + num_stages to saturate per-SM DRAM bandwidth.

C is compile-time constant (512) enabling full Triton loop pipelining.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # ── num_warps=8 (more elements/thread, higher per-thread MLP) ──
        triton.Config({}, num_warps=8,  num_stages=4),
        triton.Config({}, num_warps=8,  num_stages=6),
        triton.Config({}, num_warps=8,  num_stages=8),
        # ── num_warps=16 sweep ──
        triton.Config({}, num_warps=16, num_stages=2),
        triton.Config({}, num_warps=16, num_stages=4),
        triton.Config({}, num_warps=16, num_stages=6),
        triton.Config({}, num_warps=16, num_stages=8),
        triton.Config({}, num_warps=16, num_stages=10),
        # ── num_warps=32 sweep ── (most concurrent DRAM requests)
        triton.Config({}, num_warps=32, num_stages=2),
        triton.Config({}, num_warps=32, num_stages=4),
        triton.Config({}, num_warps=32, num_stages=6),
        triton.Config({}, num_warps=32, num_stages=8),
        triton.Config({}, num_warps=32, num_stages=10),
    ],
    key=['B', 'C'],
)
@triton.jit
def _fused_conv1x1_softmax_kernel(
    x_ptr,    # input  [B, C, HW]  contiguous NCHW
    w_ptr,    # weight [C]
    b_ptr,    # bias   [1]
    out_ptr,  # output [B, HW]
    B,
    C: tl.constexpr,        # compile-time, always 512
    HW: tl.constexpr,       # compile-time, always 4096
    IS_FP16:  tl.constexpr,
    IS_BF16:  tl.constexpr,
):
    """
    BLOCK_C=1 implicit: one scalar weight × one spatial-vector per loop step.
    No cross-warp reduction → pure scalar-vector FMA per iteration.
    C=512 is constexpr → Triton can fully pipeline the 512-iter loop.
    """
    pid = tl.program_id(0)
    hw  = tl.arange(0, HW)

    acc = tl.zeros([HW], dtype=tl.float32)

    # 512-iter scalar-vector dot product (perfectly coalesced per iteration)
    for c in range(C):
        w_c = tl.load(w_ptr + c).to(tl.float32)                          # scalar
        x_c = tl.load(x_ptr + pid * C * HW + c * HW + hw).to(tl.float32) # [HW]
        acc += w_c * x_c                                                   # FMA

    acc += tl.load(b_ptr).to(tl.float32)

    # In-kernel numerically-stable softmax
    m = tl.max(acc, axis=0)
    e = tl.exp(acc - m)
    s = tl.sum(e, axis=0)
    v = e / s

    if IS_FP16:
        tl.store(out_ptr + pid * HW + hw, v.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + pid * HW + hw, v.to(tl.bfloat16))
    else:
        tl.store(out_ptr + pid * HW + hw, v)


# ─── Two-kernel variant: conv with 2D grid + separate softmax ─────────────────
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=2, num_stages=3),
        triton.Config({}, num_warps=2, num_stages=5),
        triton.Config({}, num_warps=2, num_stages=8),
        triton.Config({}, num_warps=4, num_stages=3),
        triton.Config({}, num_warps=4, num_stages=5),
    ],
    key=['B', 'C'],
)
@triton.jit
def _conv1x1_bw64_kernel(
    x_ptr, w_ptr, b_ptr, tmp_ptr,
    B,
    C:  tl.constexpr,   # 512
    HW: tl.constexpr,   # 4096
):
    """2D grid (B, 64): 64 spatial tiles per batch → activates 28 SMs for B=1."""
    BHW: tl.constexpr = 64   # spatial tile width (must match grid dim-1 × 64)
    pid_b  = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw = pid_hw * BHW + tl.arange(0, BHW)
    acc = tl.zeros([BHW], dtype=tl.float32)

    for c in range(C):
        w_c = tl.load(w_ptr + c).to(tl.float32)
        x_c = tl.load(x_ptr + pid_b * C * HW + c * HW + hw).to(tl.float32)
        acc += w_c * x_c

    acc += tl.load(b_ptr).to(tl.float32)
    tl.store(tmp_ptr + pid_b * HW + hw, acc)


@triton.jit
def _softmax2_kernel(
    tmp_ptr, out_ptr,
    B, HW: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid = tl.program_id(0)
    hw  = tl.arange(0, HW)
    acc = tl.load(tmp_ptr + pid * HW + hw)
    m = tl.max(acc, axis=0)
    e = tl.exp(acc - m)
    s = tl.sum(e, axis=0)
    v = e / s
    if IS_FP16:
        tl.store(out_ptr + pid * HW + hw, v.to(tl.float16))
    elif IS_BF16:
        tl.store(out_ptr + pid * HW + hw, v.to(tl.bfloat16))
    else:
        tl.store(out_ptr + pid * HW + hw, v)


@torch.fx.wrap
def fused_conv1x1_softmax(in_0, in_1, in_2):
    B  = in_2.shape[0]
    C  = in_2.shape[1]
    H  = in_2.shape[2]
    W  = in_2.shape[3]
    HW = H * W   # 4096

    IS_FP16 = in_2.dtype == torch.float16
    IS_BF16 = in_2.dtype == torch.bfloat16

    # Single-kernel: fused conv+softmax, 1 CTA per batch
    # Inline softmax avoids temp buffer & second kernel launch overhead.
    out = torch.empty((B, HW), dtype=in_2.dtype, device=in_2.device)
    _fused_conv1x1_softmax_kernel[(B,)](
        in_2, in_1, in_0, out,
        B, C=C, HW=HW,
        IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return out.view(B, 1, HW)