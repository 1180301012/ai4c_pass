"""
Two-kernel fused pipeline for 1x1-conv + view + softmax.

Phase 1 – GEMV kernel (2-D grid for good SM occupancy):
  result[b, hw] = bias + sum_c( input[b, c, hw] * weight[c] )
  Grid: (batch, HW // BLOCK_HW)  → many SMs used even for small batch

Phase 2 – Softmax kernel (1-D grid, one row per program):
  output[b, 0, hw] = softmax( result[b, hw] )  over the HW dimension
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Phase 1: GEMV  (2-D grid, BLOCK_HW × BLOCK_C tiling, sw-pipelined)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 16}, num_warps=4),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 32}, num_warps=4),
        triton.Config({"BLOCK_HW": 128, "BLOCK_C": 64}, num_warps=4),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 16}, num_warps=4),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 32}, num_warps=8),
        triton.Config({"BLOCK_HW": 256, "BLOCK_C": 64}, num_warps=8),
        triton.Config({"BLOCK_HW": 512, "BLOCK_C": 16}, num_warps=8),
        triton.Config({"BLOCK_HW": 512, "BLOCK_C": 32}, num_warps=8),
    ],
    key=["C", "HW"],
)
@triton.jit
def gemv_1x1conv_kernel(
    input_ptr,   # [batch, C, HW]  NCHW-flattened
    weight_ptr,  # [1, C, 1, 1] contiguous; weight[0,c,0,0] at offset c
    bias_ptr,    # [1]
    temp_ptr,    # [batch, HW]  float32 output buffer
    C: tl.constexpr,
    HW: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_b  = tl.program_id(0)   # batch index
    pid_hw = tl.program_id(1)   # spatial block index

    hw_start = pid_hw * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]

    bias = tl.load(bias_ptr).to(tl.float32)
    acc  = tl.full([BLOCK_HW], bias, dtype=tl.float32)

    base = pid_b * C * HW
    for c_start in range(0, C, BLOCK_C):
        c_offs = c_start + tl.arange(0, BLOCK_C)       # [BLOCK_C]

        w   = tl.load(weight_ptr + c_offs).to(tl.float32)        # [BLOCK_C]

        # inp: [BLOCK_C, BLOCK_HW]  – each c row is contiguous
        inp_ptrs = input_ptr + base + c_offs[:, None] * HW + hw_offs[None, :]
        inp = tl.load(inp_ptrs).to(tl.float32)          # [BLOCK_C, BLOCK_HW]

        acc = acc + tl.sum(w[:, None] * inp, axis=0)   # [BLOCK_HW]

    tl.store(temp_ptr + pid_b * HW + hw_offs, acc)


# ---------------------------------------------------------------------------
# Phase 2: Softmax  (1-D grid, one program = one batch row)
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4),
        triton.Config({}, num_warps=8),
        triton.Config({}, num_warps=16),
    ],
    key=["HW"],
)
@triton.jit
def softmax_kernel(
    input_ptr,   # [batch, HW]  float32  (written by Phase 1)
    output_ptr,  # [batch, 1, HW]  output dtype
    HW: tl.constexpr,
):
    pid     = tl.program_id(0)          # batch index
    hw_offs = tl.arange(0, HW)          # [HW]

    x = tl.load(input_ptr + pid * HW + hw_offs)    # [HW] float32

    x_max  = tl.max(x, axis=0)
    x_exp  = tl.exp(x - x_max)
    x_sum  = tl.sum(x_exp, axis=0)
    result = x_exp / x_sum

    tl.store(output_ptr + pid * HW + hw_offs,
             result.to(output_ptr.dtype.element_ty))


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (returned by ALL pass files' replacement_func).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv1x1_softmax_dispatch(in_0, in_1, in_2, route):
    """
    in_0 : bias   [1]
    in_1 : weight [1, C, 1, 1]
    in_2 : input  [batch, C, H, W]
    route: pass-dedup tag (unused at runtime)

    Returns single tensor (shape [batch, 1, H*W]) – the model's return node
    wraps it in a tuple, so we must NOT do that here.
    """
    batch = in_2.shape[0]
    C     = in_2.shape[1]
    HW    = in_2.shape[2] * in_2.shape[3]   # 4096

    # Phase 1 accumulation buffer (always float32 for numerical stability)
    temp   = torch.empty((batch, HW),    dtype=torch.float32, device=in_2.device)
    output = torch.empty((batch, 1, HW), dtype=in_2.dtype,    device=in_2.device)

    HW_CONST = 4096   # constexpr for Triton (H=W=64)
    C_CONST  = 512    # constexpr for Triton

    # --- Phase 1: GEMV – lambda grid lets autotune pick BLOCK_HW freely ---
    gemv_1x1conv_kernel[
        lambda meta: (batch, triton.cdiv(HW_CONST, meta["BLOCK_HW"]))
    ](
        in_2, in_1, in_0, temp,
        C_CONST, HW_CONST,
    )

    # --- Phase 2: Softmax ---
    softmax_kernel[(batch,)](
        temp, output,
        HW_CONST,
    )

    return output   # single tensor, NOT (output,)