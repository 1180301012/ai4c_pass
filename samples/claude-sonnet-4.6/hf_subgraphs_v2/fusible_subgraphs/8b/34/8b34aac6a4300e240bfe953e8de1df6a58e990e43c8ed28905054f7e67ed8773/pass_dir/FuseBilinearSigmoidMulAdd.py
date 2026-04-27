import torch
import triton
import triton.language as tl


# -----------------------------------------------------------------------
# Pattern: everything after conv2d
#   Branch A: interpolate(in_4, 64x64, bilinear) -> sigmoid -> in_3 * result
#   Branch B: sigmoid(conv2d_result) -> in_2 * result -> interpolate(64x64, bilinear)
#   Output  : branch_A + branch_B
# -----------------------------------------------------------------------
def pattern(conv2d_result, in_2, in_3, in_4):
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(conv2d_result)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv2d_result, in_2, in_3, in_4):
    return (conv2d_result, in_2, in_3, in_4)


# -----------------------------------------------------------------------
# Triton kernel
#   Inputs :
#     conv_ptr  – [B, C, IH, IW]  conv2d output
#     in2_ptr   – [B, C, IH, IW]  "detail" feature
#     in3_ptr   – [B, C, OH, OW]  "context" feature (64x64)
#     in4_ptr   – [B, C, IH, IW]  attention input
#     out_ptr   – [B, C, OH, OW]  result
#
#   Fixed spatial dims: IH=IW=16, OH=OW=64 (align_corners=False, scale=0.25)
# -----------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['B', 'C'],
)
@triton.jit
def fused_bga_kernel(
    conv_ptr, in2_ptr, in3_ptr, in4_ptr, out_ptr,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    # Fixed spatial dimensions
    IH: tl.constexpr = 16
    IW: tl.constexpr = 16
    OH: tl.constexpr = 64
    OW: tl.constexpr = 64

    N_TOTAL = B * C * OH * OW

    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N_TOTAL

    # ---- decode flat index → (n, c, oh, ow) ----------------------------
    oh_ow = offsets % (OH * OW)
    nc    = offsets // (OH * OW)
    oh    = oh_ow // OW
    ow    = oh_ow % OW
    n     = nc // C
    c     = nc % C

    # ---- bilinear source coordinates (align_corners=False, scale=1/4) --
    # in_y = (oh + 0.5) * (16/64) - 0.5 = oh * 0.25 - 0.375
    in_y = oh.to(tl.float32) * 0.25 - 0.375
    in_x = ow.to(tl.float32) * 0.25 - 0.375

    # Safe floor for potentially negative values
    y_trunc = in_y.to(tl.int32)
    x_trunc = in_x.to(tl.int32)
    y_floor = tl.where(in_y < y_trunc.to(tl.float32), y_trunc - 1, y_trunc)
    x_floor = tl.where(in_x < x_trunc.to(tl.float32), x_trunc - 1, x_trunc)

    # Bilinear weights
    wy1 = in_y - y_floor.to(tl.float32)
    wx1 = in_x - x_floor.to(tl.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # Clamped indices into 16×16 input
    y0 = tl.maximum(y_floor,     0)
    y1 = tl.minimum(y_floor + 1, IH - 1)
    x0 = tl.maximum(x_floor,     0)
    x1 = tl.minimum(x_floor + 1, IW - 1)

    # Base strides
    base_16 = (n * C + c) * (IH * IW)   # offset to (n,c) plane in 16×16
    base_64 = (n * C + c) * (OH * OW)   # offset to (n,c) plane in 64×64

    # ---- Branch A: bilinear(in_4) → sigmoid → in_3 * result ------------
    in4_00 = tl.load(in4_ptr + base_16 + y0 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in4_01 = tl.load(in4_ptr + base_16 + y0 * IW + x1, mask=mask, other=0.0).to(tl.float32)
    in4_10 = tl.load(in4_ptr + base_16 + y1 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in4_11 = tl.load(in4_ptr + base_16 + y1 * IW + x1, mask=mask, other=0.0).to(tl.float32)

    in4_interp = (wy0 * wx0 * in4_00
                + wy0 * wx1 * in4_01
                + wy1 * wx0 * in4_10
                + wy1 * wx1 * in4_11)
    sig_in4 = tl.sigmoid(in4_interp)

    in3_val  = tl.load(in3_ptr + base_64 + oh * OW + ow, mask=mask, other=0.0).to(tl.float32)
    branch_a = in3_val * sig_in4

    # ---- Branch B: [sigmoid(conv)*in_2 at 4 corners] → bilinear interp -
    conv_00 = tl.load(conv_ptr + base_16 + y0 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    conv_01 = tl.load(conv_ptr + base_16 + y0 * IW + x1, mask=mask, other=0.0).to(tl.float32)
    conv_10 = tl.load(conv_ptr + base_16 + y1 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    conv_11 = tl.load(conv_ptr + base_16 + y1 * IW + x1, mask=mask, other=0.0).to(tl.float32)

    in2_00 = tl.load(in2_ptr + base_16 + y0 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in2_01 = tl.load(in2_ptr + base_16 + y0 * IW + x1, mask=mask, other=0.0).to(tl.float32)
    in2_10 = tl.load(in2_ptr + base_16 + y1 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in2_11 = tl.load(in2_ptr + base_16 + y1 * IW + x1, mask=mask, other=0.0).to(tl.float32)

    val_00 = in2_00 * tl.sigmoid(conv_00)
    val_01 = in2_01 * tl.sigmoid(conv_01)
    val_10 = in2_10 * tl.sigmoid(conv_10)
    val_11 = in2_11 * tl.sigmoid(conv_11)

    branch_b = (wy0 * wx0 * val_00
              + wy0 * wx1 * val_01
              + wy1 * wx0 * val_10
              + wy1 * wx1 * val_11)

    # ---- Final add and store (cast back to output dtype) ----------------
    result = branch_a + branch_b

    # out_ptr has the same dtype as in_3; cast via the pointer element type
    tl.store(out_ptr + base_64 + oh * OW + ow,
             result.to(out_ptr.dtype.element_ty),
             mask=mask)


# -----------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# -----------------------------------------------------------------------
@torch.fx.wrap
def fused_interp_sigmoid_mul_add(conv2d_result, in_2, in_3, in_4):
    B = in_3.shape[0]
    C = in_3.shape[1]

    out = torch.empty_like(in_3)   # [B, C, 64, 64], same dtype as in_3

    N_TOTAL = B * C * 64 * 64
    grid = lambda meta: ((N_TOTAL + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_bga_kernel[grid](
        conv2d_result, in_2, in_3, in_4, out,
        B, C,
    )
    return out


# -----------------------------------------------------------------------
# Replacement hook
# -----------------------------------------------------------------------
def replacement_func():
    return fused_interp_sigmoid_mul_add