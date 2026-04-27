import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Register torch.nn.functional.interpolate as an FX leaf function so the
# pattern function can be symbolically traced without hitting the
# "if input.dim() == ..." branch inside interpolate.
# ---------------------------------------------------------------------------
from torch.nn.functional import interpolate   # bring into module globals
torch.fx.wrap('interpolate')                  # register as leaf for FX tracing


# =============================================================================
# PATTERN: everything AFTER conv2d
#
#   Branch A: interpolate(in_4, 64×64) → sigmoid → in_3 * result
#   Branch B: sigmoid(conv_result) → in_2 * result → interpolate(64×64)
#   Output  : branch_A + branch_B
# =============================================================================
def pattern(conv_result, in_2, in_3, in_4):
    tmp_3 = interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(conv_result)
    tmp_7 = in_2 * tmp_6
    tmp_8 = interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(conv_result, in_2, in_3, in_4):
    return (conv_result, in_2, in_3, in_4)


# =============================================================================
# KERNEL: Fused bilinear-upsample + sigmoid + mul + add
#
# For each output (n, c, oh, ow) in [B, C, 64, 64]:
#
#   Branch A (all in-register):
#     interp_in4  = bilinear_interp(in4[n,c], oh, ow)    [16×16 → 64×64]
#     branch_A    = in3[n,c,oh,ow] * sigmoid(interp_in4)
#
#   Branch B (compute pre-interp values at the 4 corners):
#     val_yx = in2[n,c,y,x] * sigmoid(conv[n,c,y,x])   for each corner (y,x)
#     branch_B = bilinear_interp over those 4 val_yx
#
#   output[n,c,oh,ow] = branch_A + branch_B
#
# Fixed spatial dims: IH=IW=16 → OH=OW=64, align_corners=False, scale=0.25
# =============================================================================
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
def fused_bga_v2_kernel(
    conv_ptr, in2_ptr, in3_ptr, in4_ptr, out_ptr,
    B, C,
    BLOCK_SIZE: tl.constexpr,
):
    IH: tl.constexpr = 16
    IW: tl.constexpr = 16
    OH: tl.constexpr = 64
    OW: tl.constexpr = 64

    N_TOTAL = B * C * OH * OW

    pid     = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < N_TOTAL

    # ---- decode flat index → (n, c, oh, ow) --------------------------------
    oh_ow = offsets % (OH * OW)
    nc    = offsets // (OH * OW)
    oh    = oh_ow // OW
    ow    = oh_ow % OW
    n     = nc // C
    c     = nc % C

    # ---- bilinear source coordinates (align_corners=False, scale=0.25) -----
    # in_y = (oh + 0.5) * (16/64) - 0.5 = oh * 0.25 - 0.375
    in_y = oh.to(tl.float32) * 0.25 - 0.375
    in_x = ow.to(tl.float32) * 0.25 - 0.375

    # Safe floor: Triton's .to(int32) truncates toward zero; correct negatives
    y_trunc = in_y.to(tl.int32)
    x_trunc = in_x.to(tl.int32)
    y_floor = tl.where(in_y < y_trunc.to(tl.float32), y_trunc - 1, y_trunc)
    x_floor = tl.where(in_x < x_trunc.to(tl.float32), x_trunc - 1, x_trunc)

    wy1 = in_y - y_floor.to(tl.float32)
    wx1 = in_x - x_floor.to(tl.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # Clamped indices into the 16×16 input
    y0 = tl.maximum(y_floor,     0)
    y1 = tl.minimum(y_floor + 1, IH - 1)
    x0 = tl.maximum(x_floor,     0)
    x1 = tl.minimum(x_floor + 1, IW - 1)

    base_16 = (n * C + c) * (IH * IW)
    base_64 = (n * C + c) * (OH * OW)

    # ---- Branch A: bilinear(in4) → sigmoid → × in3 -------------------------
    in4_00 = tl.load(in4_ptr + base_16 + y0 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in4_01 = tl.load(in4_ptr + base_16 + y0 * IW + x1, mask=mask, other=0.0).to(tl.float32)
    in4_10 = tl.load(in4_ptr + base_16 + y1 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in4_11 = tl.load(in4_ptr + base_16 + y1 * IW + x1, mask=mask, other=0.0).to(tl.float32)

    in4_blerp = (wy0 * wx0 * in4_00 + wy0 * wx1 * in4_01
               + wy1 * wx0 * in4_10 + wy1 * wx1 * in4_11)
    sig_in4 = tl.sigmoid(in4_blerp)

    in3_val  = tl.load(in3_ptr + base_64 + oh * OW + ow, mask=mask, other=0.0).to(tl.float32)
    branch_a = in3_val * sig_in4

    # ---- Branch B: [sigmoid(conv) × in2] at 4 corners → bilinear interp ---
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

    branch_b = (wy0 * wx0 * val_00 + wy0 * wx1 * val_01
              + wy1 * wx0 * val_10 + wy1 * wx1 * val_11)

    # ---- Store result --------------------------------------------------------
    result = branch_a + branch_b
    tl.store(
        out_ptr + base_64 + oh * OW + ow,
        result.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


# =============================================================================
# WRAPPER
# =============================================================================
@torch.fx.wrap
def fused_interp_sigmoid_mul_add_v2(conv_result, in_2, in_3, in_4):
    B = in_3.shape[0]
    C = in_3.shape[1]

    out     = torch.empty_like(in_3)   # [B, C, 64, 64], same dtype
    N_TOTAL = B * C * 64 * 64

    grid = lambda meta: ((N_TOTAL + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_bga_v2_kernel[grid](
        conv_result, in_2, in_3, in_4, out,
        B, C,
    )
    return out


def replacement_func():
    return fused_interp_sigmoid_mul_add_v2