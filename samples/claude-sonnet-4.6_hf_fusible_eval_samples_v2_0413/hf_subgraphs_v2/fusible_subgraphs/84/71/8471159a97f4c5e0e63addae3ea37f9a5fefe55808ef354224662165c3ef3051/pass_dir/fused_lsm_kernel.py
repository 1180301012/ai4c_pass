"""
Shared Triton kernel: fused linear + sigmoid + broadcast multiply.

For each (b, c) pair:
  gate = sigmoid( dot(in_2[b,:], in_1[c,:]) + in_0[c] )
  out[b, c, :] = in_3[b, c, :] * gate

in_0 : bias    [C]
in_1 : weight  [C, K]
in_2 : input   [B, K]
in_3 : feature [B, C, H, W]
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=1),   # 128-bit fp16, max occupancy: 64 blocks/SM
        triton.Config({'BLOCK_HW': 512},  num_warps=2),   # 128-bit fp16, many programs for B=32
        triton.Config({'BLOCK_HW': 512},  num_warps=4),   # 128-bit fp32
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),   # 128-bit fp16
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),   # 128-bit fp32
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),   # 128-bit fp16/bf16 for B=128
    ],
    key=['B', 'HW'],
    warmup=5,
    rep=5,
)
@triton.jit
def _fused_lsm_kernel(
    in_0_ptr,                    # bias    [C]
    in_1_ptr,                    # weight  [C, K]
    in_2_ptr,                    # input   [B, K]
    in_3_ptr,                    # feature [B, C, HW] (contiguous)
    out_ptr,                     # output  [B, C, HW] (contiguous)
    B,
    C: tl.constexpr,
    K: tl.constexpr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    # 3D grid: axis-0 = b, axis-1 = c, axis-2 = hw block
    # Avoids expensive integer division bc_id // C and bc_id % C
    b          = tl.program_id(0)
    c          = tl.program_id(1)
    hw_blk_id  = tl.program_id(2)

    # ---- Compute gate: sigmoid(in_2[b] · in_1[c] + in_0[c]) ----
    k_offs  = tl.arange(0, K)
    in2     = tl.load(in_2_ptr + b * K + k_offs)
    in1     = tl.load(in_1_ptr + c * K + k_offs)
    dot     = tl.sum(in2.to(tl.float32) * in1.to(tl.float32), axis=0)
    bias    = tl.load(in_0_ptr + c).to(tl.float32)
    gate    = tl.sigmoid(dot + bias)

    # ---- Apply gate to the feature block ----
    hw_offs = hw_blk_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW

    base  = (b * C + c) * HW
    x     = tl.load(in_3_ptr + base + hw_offs, mask=hw_mask, other=0.0)
    y     = x.to(tl.float32) * gate
    tl.store(out_ptr + base + hw_offs, y.to(x.dtype), mask=hw_mask)


@triton.jit
def _fused_lsm_kernel_noa(
    in_0_ptr,
    in_1_ptr,
    in_2_ptr,
    in_3_ptr,
    out_ptr,
    B,
    C: tl.constexpr,
    K: tl.constexpr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    """Same kernel body as _fused_lsm_kernel but WITHOUT @triton.autotune.
    JIT-compiles once on first call (during warmup), no re-autotuning in trials.
    """
    b          = tl.program_id(0)
    c          = tl.program_id(1)
    hw_blk_id  = tl.program_id(2)

    k_offs  = tl.arange(0, K)
    in2     = tl.load(in_2_ptr + b * K + k_offs)
    in1     = tl.load(in_1_ptr + c * K + k_offs)
    dot     = tl.sum(in2.to(tl.float32) * in1.to(tl.float32), axis=0)
    bias    = tl.load(in_0_ptr + c).to(tl.float32)
    gate    = tl.sigmoid(dot + bias)

    hw_offs = hw_blk_id * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_offs < HW
    base  = (b * C + c) * HW
    x     = tl.load(in_3_ptr + base + hw_offs, mask=hw_mask, other=0.0)
    y     = x.to(tl.float32) * gate
    tl.store(out_ptr + base + hw_offs, y.to(x.dtype), mask=hw_mask)


@torch.fx.wrap
def fused_linear_sigmoid_mul(in_0, in_1, in_2, in_3):
    """
    Drop-in replacement for:
        linear = F.linear(in_2, in_1, in_0)          # [B, C]
        gate   = sigmoid(linear).view(B, C, 1, 1)    # [B, C, 1, 1]
        out    = in_3 * gate                          # [B, C, H, W]
    Manual dispatch: no autotune overhead, stable trial timing.
    """
    B  = in_2.shape[0]
    C  = in_1.shape[0]   # 64
    K  = in_1.shape[1]   # 8
    HW = in_3.shape[2] * in_3.shape[3]

    out = torch.empty_like(in_3)

    # Empirically-tuned config selection for 128-bit vectorized loads:
    #   fp16/bf16: BLOCK_HW/num_threads = 8  (e.g. BLOCK_HW=256, warps=1)
    #   fp32:      BLOCK_HW/num_threads = 4  (e.g. BLOCK_HW=1024, warps=8)
    if B <= 32:
        # B=1 and B=32: max occupancy (64 blocks/SM), 128-bit fp16/bf16
        BHW, NW = 256, 1
    elif HW <= 4000:
        # B=128, small HW (3136): fp16/bf16 dominated, 128-bit loads
        BHW, NW = 2048, 8
    else:
        # B=128, large HW (4096+): likely fp32, 128-bit fp32 loads
        BHW, NW = 1024, 8

    _fused_lsm_kernel_noa[(B, C, triton.cdiv(HW, BHW))](
        in_0, in_1, in_2, in_3, out,
        B, C, K, HW,
        BLOCK_HW=BHW,
        num_warps=NW,
    )

    return out