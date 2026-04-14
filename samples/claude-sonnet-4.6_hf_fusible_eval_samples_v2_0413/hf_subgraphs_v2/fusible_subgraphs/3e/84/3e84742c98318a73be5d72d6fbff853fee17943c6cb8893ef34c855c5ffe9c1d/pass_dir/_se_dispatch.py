"""
Shared Triton kernel + dispatch wrapper for SE HardSigmoid + broadcast-multiply fusion.
Both FuseConvHardSigmoidMul_v1 and FuseConvHardSigmoidMul_v2 import from here so that
replacement_func() returns the *same* Python object in both passes, satisfying the
output_pass_replacement_func_limit constraint.

Design:
- 2-D autotuned grid  (B*Cout, ceil(HW/BLOCK_HW)).
- 7 configs; autotune selects best BLOCK_HW/num_warps per (B, Cout, HW).
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},   num_warps=2),
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['B', 'Cout', 'HW'],
)
@triton.jit
def _hardsigmoid_broadcast_mul_kernel(
    attn_ptr,   # [B, Cout, 1, 1]  conv2d output  (stride [Cout,1,1,1])
    in2_ptr,    # [B, Cout, H, W]  feature map    (contiguous NCHW)
    out_ptr,    # [B, Cout, H, W]  output
    B, Cout, HW,
    ADD_VAL: tl.constexpr,
    DIV_VAL: tl.constexpr,
    BLOCK_HW:  tl.constexpr,
):
    # 2-D grid: dim0 = B*Cout (flat), dim1 = HW tile
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b     = pid_bc // Cout
    c     = pid_bc  % Cout

    # ----------------------------------------------------------------
    # 1. Load the per-channel scalar from attn [B, Cout, 1, 1]
    #    Element [b,c,0,0] is at flat offset  b*Cout + c
    # ----------------------------------------------------------------
    attn_raw = tl.load(attn_ptr + b * Cout + c).to(tl.float32)

    # ----------------------------------------------------------------
    # 2. HardSigmoid:  clamp((x + ADD_VAL) / DIV_VAL, 0, 1)
    # ----------------------------------------------------------------
    attn = tl.minimum(tl.maximum((attn_raw + ADD_VAL) / DIV_VAL, 0.0), 1.0)

    # ----------------------------------------------------------------
    # 3. Scale the HW tile of in_2[b, c, :]
    # ----------------------------------------------------------------
    hw_start = pid_hw * BLOCK_HW
    base     = (b * Cout + c) * HW + hw_start
    hw_offs  = tl.arange(0, BLOCK_HW)
    hw_mask  = (hw_start + hw_offs) < HW

    x2  = tl.load(in2_ptr + base + hw_offs, mask=hw_mask, other=0.0)
    out = (x2.to(tl.float32) * attn).to(x2.dtype)
    tl.store(out_ptr + base + hw_offs, out, mask=hw_mask)


@torch.fx.wrap
def se_hardsigmoid_mul_dispatch(attn, in_2, route):
    """
    Shared dispatch entry used by both v1 and v2 passes.

    attn  : [B, Cout, 1, 1]  – conv2d output (already computed by cuBLAS)
    in_2  : [B, Cout, H, W]  – feature map to scale
    route : "v1"  → HardSigmoid (x+1)/2
            "v2"  → HardSigmoid (x+3)/6
    """
    B    = in_2.shape[0]
    Cout = in_2.shape[1]
    HW   = in_2.shape[2] * in_2.shape[3]

    out  = torch.empty_like(in_2)
    grid = lambda meta: (B * Cout, triton.cdiv(HW, meta['BLOCK_HW']))

    if route == "v1":
        _hardsigmoid_broadcast_mul_kernel[grid](
            attn, in_2, out, B, Cout, HW,
            ADD_VAL=1.0, DIV_VAL=2.0,
        )
    elif route == "v2":
        _hardsigmoid_broadcast_mul_kernel[grid](
            attn, in_2, out, B, Cout, HW,
            ADD_VAL=3.0, DIV_VAL=6.0,
        )

    return out