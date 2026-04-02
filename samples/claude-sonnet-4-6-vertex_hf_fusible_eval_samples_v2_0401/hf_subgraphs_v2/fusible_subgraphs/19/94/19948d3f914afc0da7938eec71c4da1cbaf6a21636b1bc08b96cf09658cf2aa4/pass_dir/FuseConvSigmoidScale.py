import torch
import triton
import triton.language as tl


# -------------------------------------------------------------------------
# Pattern: grouped conv2d (1x1, groups=4) + sigmoid + view + mul + contiguous
# -------------------------------------------------------------------------
def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# -------------------------------------------------------------------------
# Single fused Triton kernel — 2D grid (C_out × HW_blocks)
#
# Each program handles ONE output channel (c) and ONE tile of HW pixels.
# It computes the grouped 1×1 conv + sigmoid inline (17 tiny scalar loads,
# all served from L1 cache), then scales a contiguous tile of the feature map.
# This replaces 3 separate GPU kernels (conv + sigmoid + multiply) with 1,
# saving ~2 kernel-launch round-trips (~20μs on A30).
#
# Grid: (C_out=96, ceil(HW / BLOCK_HW))
# With 864–2400 programs depending on HW, all 28 A30 SMs stay busy.
# -------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 128},  num_warps=2),
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def fused_conv_sigmoid_scale_kernel(
    in3_ptr,    # [1, 32, 1, 1]  gap input  (contiguous → flat index = channel)
    w_ptr,      # [96, 8, 1, 1]  conv weight (contiguous → flat index = c*8 + k)
    b_ptr,      # [96]           conv bias
    in2_ptr,    # [1, 96, H, W]  feature map (contiguous → flat index = c*HW + hw)
    out_ptr,    # [1, 96, H, W]  output
    HW,         # H * W  (runtime, varies across test cases)
    BLOCK_HW: tl.constexpr,
):
    c        = tl.program_id(0)   # output channel index  (0 … 95)
    hw_block = tl.program_id(1)   # tile index over HW

    # ---- grouped conv2d (groups=4, C_in=32, C_out=96) ----
    # C_in_per_group = 8,  C_out_per_group = 24
    g        = c // 24
    in3_base = g * 8
    w_base   = c * 8

    # Accumulate in float32; 8 values from 32-element / 768-element tensors
    # → always served from L1 cache across all hw_block repetitions.
    bias = tl.load(b_ptr + c).to(tl.float32)

    i0 = tl.load(in3_ptr + in3_base + 0).to(tl.float32) * tl.load(w_ptr + w_base + 0).to(tl.float32)
    i1 = tl.load(in3_ptr + in3_base + 1).to(tl.float32) * tl.load(w_ptr + w_base + 1).to(tl.float32)
    i2 = tl.load(in3_ptr + in3_base + 2).to(tl.float32) * tl.load(w_ptr + w_base + 2).to(tl.float32)
    i3 = tl.load(in3_ptr + in3_base + 3).to(tl.float32) * tl.load(w_ptr + w_base + 3).to(tl.float32)
    i4 = tl.load(in3_ptr + in3_base + 4).to(tl.float32) * tl.load(w_ptr + w_base + 4).to(tl.float32)
    i5 = tl.load(in3_ptr + in3_base + 5).to(tl.float32) * tl.load(w_ptr + w_base + 5).to(tl.float32)
    i6 = tl.load(in3_ptr + in3_base + 6).to(tl.float32) * tl.load(w_ptr + w_base + 6).to(tl.float32)
    i7 = tl.load(in3_ptr + in3_base + 7).to(tl.float32) * tl.load(w_ptr + w_base + 7).to(tl.float32)

    conv_out = bias + i0 + i1 + i2 + i3 + i4 + i5 + i6 + i7
    scale    = tl.sigmoid(conv_out)   # float32

    # ---- scale a contiguous tile of the feature map ----
    hw_start = hw_block * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    mask     = hw_offs < HW

    base     = c * HW
    in2_vals = tl.load(in2_ptr + base + hw_offs, mask=mask)

    # float32 multiply → cast back to original dtype
    result = (in2_vals.to(tl.float32) * scale).to(in2_vals.dtype)
    tl.store(out_ptr + base + hw_offs, result, mask=mask)


# -------------------------------------------------------------------------
# Wrapper (must be decorated with @torch.fx.wrap)
# -------------------------------------------------------------------------
@torch.fx.wrap
def fused_conv_sigmoid_scale(in_0, in_1, in_2, in_3):
    """
    Fused replacement for:
        conv2d(in_3, in_1, in_0, stride=1, pad=0, dil=1, groups=4)
        → sigmoid → view(1,-1,1,1) → in_2 * scale → contiguous

    2D grid of 96 × ceil(HW/BLOCK_HW) programs maximises A30 SM utilisation.
    Replaces 3 GPU kernel launches with 1.

    in_0 : [96]          bias
    in_1 : [96, 8, 1, 1] conv weight
    in_2 : [1, 96, H, W] feature map to scale
    in_3 : [1, 32, 1, 1] gap output fed into conv
    """
    C_out = in_2.shape[1]                  # 96
    HW    = in_2.shape[2] * in_2.shape[3]

    in3_c = in_3.contiguous()
    in1_c = in_1.contiguous()
    in2_c = in_2.contiguous()

    out  = torch.empty_like(in2_c)
    grid = lambda meta: (C_out, triton.cdiv(HW, meta['BLOCK_HW']))

    fused_conv_sigmoid_scale_kernel[grid](
        in3_c, in1_c, in_0, in2_c, out,
        HW,
    )

    return out


def replacement_func():
    return fused_conv_sigmoid_scale