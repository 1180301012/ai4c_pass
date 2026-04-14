"""
Pass: Fuse conv2d(in_3,...) -> stack([x],0) -> sum(0) -> cat([result,in_2],1)
into a single fused Triton kernel (1x1 GEMM + cat).

This is the variant where:
  - in_3 is the convolution input
  - in_2 is concatenated after the conv output
"""
import torch
import triton
import triton.language as tl

# Import the shared kernels from the companion pass file
from pass_dir.FuseConv1x1StackSumCat_conv_in2_cat_in3 import (
    _conv1x1_gemm_kernel,
    _copy_cat_kernel,
)


# ─────────────────────────────────────────────────────────────────────────────
# Pattern (must mirror model.py dataflow exactly)
# ─────────────────────────────────────────────────────────────────────────────
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.stack([conv2d], dim=0)
    tmp_4 = tmp_3.sum(dim=0)
    tmp_5 = torch.cat([tmp_4, in_2], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    # Reorder so that replacement receives (bias, weight, conv_input, cat_other)
    # in_0=bias, in_1=weight, in_3=conv_input, in_2=cat_other
    return (in_0, in_1, in_3, in_2)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper  (same logic, different argument mapping)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def _fused_conv1x1_cat_b(bias, weight, conv_input, cat_other):
    """
    bias:       [C_out]
    weight:     [C_out, C_in, 1, 1]
    conv_input: [N, C_in, H, W]
    cat_other:  [N, C_y,  H, W]
    returns:    [N, C_out + C_y, H, W]
    """
    N, C_in, H, W = conv_input.shape
    C_out   = weight.shape[0]
    C_y     = cat_other.shape[1]
    C_total = C_out + C_y
    HW      = H * W
    N_M     = N * HW

    out = torch.empty((N, C_total, H, W),
                      dtype=conv_input.dtype,
                      device=conv_input.device)

    is_fp16 = conv_input.dtype == torch.float16
    is_bf16 = conv_input.dtype == torch.bfloat16

    # ── 1.  GEMM: conv1x1 → first C_out channels ──────────────────────────
    # weight [C_out, C_in, 1, 1] has stride (C_in,1,1,1); same layout as [C_out,C_in]
    # Use lambda grid so it adapts to the BLOCK_M/BLOCK_N chosen by autotune
    _nm, _co = N_M, C_out
    grid_gemm = lambda meta: (triton.cdiv(_nm, meta['BLOCK_M']),
                              triton.cdiv(_co, meta['BLOCK_N']))
    _conv1x1_gemm_kernel[grid_gemm](
        conv_input, weight, bias, out,
        N_M, C_in, C_out, C_total, HW,
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )

    # ── 2.  Copy: cat_other → last C_y channels ───────────────────────────
    COPY_BS   = 2048
    copy_grid = (triton.cdiv(N * C_y * HW, COPY_BS),)
    _copy_cat_kernel[copy_grid](
        cat_other, out,
        N, C_y, C_out, C_total, HW,
        BLOCK_SIZE=COPY_BS,
    )

    return out


def replacement_func():
    return _fused_conv1x1_cat_b