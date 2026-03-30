"""
Fused pass: conv2d (1x1, Cout=1) + view(1,1,-1) + softmax(dim=-1)
Batch size B=1, spatial HW=4096, channels C=512.

Key optimisation insight:
  For B=1 the compute is tiny and GPU occupancy is the bottleneck.
  Using BLOCK_HW=64 spawns HW/64 = 64 CTAs, spreading work across
  many SMs and saturating HBM bandwidth (vs only 4-16 CTAs with
  larger blocks that the autotuner kept selecting due to cold-cache bias).
  NO AUTOTUNE - fixed config eliminates the cold-cache config selection
  problem that made the autotuner choose large BLOCK_HW values.

Pipeline:
  1. _dotprod_kernel_b1  – grid=(1,64): computes y[b,hw]=dot(x[b,:,hw],w)+bias
  2. _softmax_kernel_b1  – grid=(1,): numerically-stable softmax over HW=4096
"""

import torch
import triton
import triton.language as tl


# ── Kernel 1: batched dot-product (1x1 conv, Cout=1) ─────────────────────────

@triton.jit
def _dotprod_kernel_b1(
    x_ptr, w_ptr, b_ptr, y_ptr,
    B, C, HW,
    BLOCK_HW: tl.constexpr,   # fixed 64
    BLOCK_C:  tl.constexpr,   # fixed 128
):
    """y[pid_b, hw_tile] = sum_c(x[pid_b,c,hw_tile] * w[c]) + bias
    
    Grid shape: (B, HW // BLOCK_HW) = (1, 64) for B=1, HW=4096.
    64 CTAs × 4 warps = 256 warps spread over 28 SMs → good HBM utilisation.
    """
    pid_b  = tl.program_id(0)
    pid_hw = tl.program_id(1)

    hw_base = pid_hw * BLOCK_HW
    hw_offs = hw_base + tl.arange(0, BLOCK_HW)

    acc = tl.zeros([BLOCK_HW], dtype=tl.float32)

    # Standard Python range → num_stages in the launch config drives pipelining
    for c_base in range(0, C, BLOCK_C):
        c_offs = c_base + tl.arange(0, BLOCK_C)
        # weight: [BLOCK_C] – tiny, stays in L2 after first CTA
        w = tl.load(w_ptr + c_offs).to(tl.float32)
        # input: [BLOCK_C, BLOCK_HW] – rows contiguous in HW dimension
        x = tl.load(
            x_ptr + pid_b * C * HW + c_offs[:, None] * HW + hw_offs[None, :]
        ).to(tl.float32)
        acc += tl.sum(x * w[:, None], axis=0)

    bias = tl.load(b_ptr).to(tl.float32)
    acc += bias
    tl.store(y_ptr + pid_b * HW + hw_offs, acc)


# ── Kernel 2: softmax over HW=4096 ───────────────────────────────────────────

@triton.jit
def _softmax_kernel_b1(
    out_ptr, inp_ptr,
    HW: tl.constexpr,
):
    """Numerically-stable single-pass softmax.
    Input is fp32 intermediate; output dtype follows out_ptr.
    32 warps (1024 threads) on 1 SM → hides latency for 16 KB.
    """
    pid  = tl.program_id(0)
    offs = tl.arange(0, HW)
    x    = tl.load(inp_ptr + pid * HW + offs)   # float32
    m    = tl.max(x, axis=0)
    e    = tl.exp(x - m)
    s    = tl.sum(e, axis=0)
    tl.store(out_ptr + pid * HW + offs, (e / s).to(out_ptr.dtype.element_ty))


# ── Wrapper ───────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv_softmax_b1(in_0, in_1, in_2):
    """
    in_0 : bias   [1]
    in_1 : weight [1, 512, 1, 1]
    in_2 : input  [1, 512, 64, 64]  (B=1 for this pass)

    Two-kernel pipeline:
      conv dotprod → float32 intermediate → Triton softmax → output dtype
    """
    B, C, H, W = in_2.shape
    HW = H * W  # = 4096

    # Float32 intermediate for numerical stability
    mid = torch.empty((B, HW), device=in_2.device, dtype=torch.float32)

    # Fixed grid: (1, 64) for B=1, HW=4096, BLOCK_HW=64
    _dotprod_kernel_b1[(B, HW // 64)](
        in_2, in_1.reshape(-1), in_0,
        mid,
        B, C, HW,
        BLOCK_HW=64,
        BLOCK_C=128,
        num_warps=4,
        num_stages=3,   # software-pipeline 3 iterations for latency hiding
    )

    out = torch.empty((B, HW), device=in_2.device, dtype=in_2.dtype)
    _softmax_kernel_b1[(B,)](
        out, mid,
        HW=4096,
        num_warps=32,   # 1024 threads for max latency hiding on single CTA
    )

    return out.view(B, 1, HW)


# ── Pattern / replacement_args / replacement_func ─────────────────────────────

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = conv2d.view(1, 1, -1)
    tmp_4  = tmp_3.softmax(dim=-1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_conv_softmax_b1