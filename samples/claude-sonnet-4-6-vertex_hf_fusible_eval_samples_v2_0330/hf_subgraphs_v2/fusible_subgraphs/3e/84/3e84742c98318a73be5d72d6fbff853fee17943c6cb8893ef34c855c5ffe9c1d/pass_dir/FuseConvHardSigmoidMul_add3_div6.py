"""
Fused optimization pass for:
  conv2d (1x1 kernel, 1x1 spatial input)
  + hard-sigmoid with (x + 3.0) / 6.0
  + SE-style channel multiply

This pattern appears in MobileNetV3 (mmseg) SE blocks (hard_sigmoid6 variant).
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern  (add=3.0, div=6.0 variant — mmseg MobileNetV3)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = conv2d + 3.0
    tmp_4 = tmp_3 / 6.0
    tmp_5 = tmp_4.clamp_(0.0, 1.0)
    tmp_6 = in_2 * tmp_5
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# ---------------------------------------------------------------------------
# Fused Triton kernel  (hard-sigmoid6 = clamp((x+3)/6, 0, 1))
# Grid: (B, C_out, ceil(HW / BLOCK_HW))  — 3D avoids integer division.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64,   'BLOCK_CIN': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 128,  'BLOCK_CIN': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 128,  'BLOCK_CIN': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256,  'BLOCK_CIN': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 512,  'BLOCK_CIN': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_CIN': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024, 'BLOCK_CIN': 128}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048, 'BLOCK_CIN': 128}, num_warps=8),
    ],
    key=['C_in', 'HW'],
)
@triton.jit
def se_hardsigmoid6_kernel(
    in3_ptr, in1_ptr, in0_ptr, in2_ptr, out_ptr,
    B, C_in, C_out, HW,
    BLOCK_HW:  tl.constexpr,
    BLOCK_CIN: tl.constexpr,
):
    b      = tl.program_id(0)
    c_out  = tl.program_id(1)
    pid_hw = tl.program_id(2)

    cin_off  = tl.arange(0, BLOCK_CIN)
    cin_mask = cin_off < C_in

    x = tl.load(in3_ptr + b * C_in + cin_off,
                mask=cin_mask, other=0.0).to(tl.float32)
    w = tl.load(in1_ptr + c_out * C_in + cin_off,
                mask=cin_mask, other=0.0).to(tl.float32)
    dot = tl.sum(x * w, axis=0)

    bias = tl.load(in0_ptr + c_out).to(tl.float32)
    dot  = dot + bias

    # hard-sigmoid: clamp((dot + 3) / 6, 0, 1)
    attn_f32 = (dot + 3.0) * 0.16666666666666666
    attn_f32 = tl.minimum(tl.maximum(attn_f32, 0.0), 1.0)

    pid_bc  = b * C_out + c_out
    hw_off  = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    hw_mask = hw_off < HW
    base    = pid_bc * HW

    in2_vals = tl.load(in2_ptr + base + hw_off, mask=hw_mask, other=0.0)
    attn     = attn_f32.to(in2_vals.dtype)
    out_vals = in2_vals * attn
    tl.store(out_ptr + base + hw_off, out_vals, mask=hw_mask)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def se_hardsigmoid6_wrapper(in_0, in_1, in_2, in_3):
    B     = in_3.shape[0]
    C_in  = in_3.shape[1]
    C_out = in_1.shape[0]
    HW    = in_2.shape[2] * in_2.shape[3]

    out = torch.empty_like(in_2)

    se_hardsigmoid6_kernel[
        lambda meta: (B, C_out, triton.cdiv(HW, meta['BLOCK_HW']))
    ](
        in_3, in_1, in_0, in_2, out,
        B, C_in, C_out, HW,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func
# ---------------------------------------------------------------------------

def replacement_func():
    return se_hardsigmoid6_wrapper