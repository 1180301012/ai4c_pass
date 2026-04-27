"""
Fuse: cat([in0,in1,in2,in3], dim=1)
      → adaptive_avg_pool2d((1,1))
      → dropout(p=0.5, training=False)   [no-op in inference]
      → flatten(1)

into a single Triton kernel that computes per-channel spatial means
directly from the four source tensors, skipping the large intermediate
cat buffer (~100 KB).

Key design decisions
====================
* ONE kernel launch (minimises dispatch overhead for this tiny workload).
* 1-D grid over C_total=2048 channels  (BLOCK_C=64 → 32 blocks → all 24 SMs busy).
* Spatial accumulation is a fully-unrolled scalar loop (tl.static_range(HW))
  so the compiler emits 25 independent 1-D loads and the hardware can
  pipeline them without any shared-memory or 2-D block overhead.
* Group membership (which source tensor each channel belongs to) is resolved
  once with tl.where on the 1-D channel vector; the masked loads for the
  "wrong" tensors become predicated no-ops.
* No autotuning overhead – fixed BLOCK_C=64, num_warps=2.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    tmp_0 = torch.cat([in_0, in_1, in_2, in_3], 1)
    tmp_1 = torch.nn.functional.adaptive_avg_pool2d(tmp_0, (1, 1))
    tmp_2 = torch.nn.functional.dropout(tmp_1, 0.5, False, False)
    tmp_3 = torch.flatten(tmp_2, 1)
    return tmp_3


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Single fused kernel
# Grid : (ceil(C_total / BLOCK_C), B)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_cat_mean_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    C0, C1, C2, C3,
    out_C_total,           # == C0+C1+C2+C3  (output batch stride)
    s_b0, s_c0,
    s_b1, s_c1,
    s_b2, s_c2,
    s_b3, s_c3,
    HW: tl.constexpr,      # spatial size H*W (= 25) — constexpr enables unrolling
    BLOCK_C: tl.constexpr,
):
    c_pid = tl.program_id(0)
    b     = tl.program_id(1)

    # 1-D channel vector for this block
    c = c_pid * BLOCK_C + tl.arange(0, BLOCK_C)
    C_total = C0 + C1 + C2 + C3
    c_mask  = c < C_total

    # ---- group membership ----
    in_g0 = c < C0
    in_g1 = (c >= C0)        & (c < C0 + C1)
    in_g2 = (c >= C0 + C1)   & (c < C0 + C1 + C2)
    in_g3 =  c >= C0 + C1 + C2

    # Clamped in-group channel indices (safe for masked loads)
    c_g0 = tl.where(in_g0, c,                  0)
    c_g1 = tl.where(in_g1, c - C0,             0)
    c_g2 = tl.where(in_g2, c - (C0 + C1),      0)
    c_g3 = tl.where(in_g3, c - (C0 + C1 + C2), 0)

    # Compound masks: in_gX & channel-in-range
    m0 = c_mask & in_g0
    m1 = c_mask & in_g1
    m2 = c_mask & in_g2
    m3 = c_mask & in_g3

    # ---- accumulate over spatial dimension (fully unrolled) ----
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)

    for hw in tl.static_range(HW):
        v0 = tl.load(in0_ptr + b * s_b0 + c_g0 * s_c0 + hw, mask=m0, other=0.0).to(tl.float32)
        v1 = tl.load(in1_ptr + b * s_b1 + c_g1 * s_c1 + hw, mask=m1, other=0.0).to(tl.float32)
        v2 = tl.load(in2_ptr + b * s_b2 + c_g2 * s_c2 + hw, mask=m2, other=0.0).to(tl.float32)
        v3 = tl.load(in3_ptr + b * s_b3 + c_g3 * s_c3 + hw, mask=m3, other=0.0).to(tl.float32)
        acc += v0 + v1 + v2 + v3

    mean = acc * (1.0 / HW)

    # ---- store to output[b, c] ----
    tl.store(out_ptr + b * out_C_total + c, mean, mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_cat_avgpool_flatten(in0, in1, in2, in3):
    B       = in0.shape[0]
    C0, C1  = in0.shape[1], in1.shape[1]
    C2, C3  = in2.shape[1], in3.shape[1]
    HW      = in0.shape[2] * in0.shape[3]   # 25 for the target workload
    C_total = C0 + C1 + C2 + C3             # 2048

    BLOCK_C = 64   # 32 blocks for B=1,C_total=2048 → fills all 24 SMs on A30

    out = torch.empty((B, C_total), dtype=in0.dtype, device=in0.device)

    _fused_cat_mean_kernel[
        (triton.cdiv(C_total, BLOCK_C), B)
    ](
        in0, in1, in2, in3,
        out,
        C0, C1, C2, C3,
        C_total,
        in0.stride(0), in0.stride(1),
        in1.stride(0), in1.stride(1),
        in2.stride(0), in2.stride(1),
        in3.stride(0), in3.stride(1),
        HW=HW,
        BLOCK_C=BLOCK_C,
        num_warps=2,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement hook
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_cat_avgpool_flatten