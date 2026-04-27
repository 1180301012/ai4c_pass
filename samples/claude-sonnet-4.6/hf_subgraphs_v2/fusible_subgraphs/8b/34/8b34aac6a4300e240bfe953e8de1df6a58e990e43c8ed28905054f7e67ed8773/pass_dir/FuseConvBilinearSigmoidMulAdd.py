import torch
import triton
import triton.language as tl


# =============================================================================
# PATTERN: Full computation (conv2d + bilinear interp + sigmoid + mul + add)
#
# Matches models with:
#   in_0: bias  [OC]
#   in_1: weight [OC, IC, 1, 1]
#   in_2: detail features [B, IC, 16, 16]
#   in_3: context features [B, IC, 64, 64]
#   in_4: attention input [B, IC, 16, 16]
#   in_5: conv input [B, IC, 16, 16]
# =============================================================================
def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    tmp_2 = torch.conv2d(in_5, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3 = torch.nn.functional.interpolate(in_4, (64, 64), None, 'bilinear', False)
    tmp_4 = torch.sigmoid(tmp_3)
    tmp_5 = in_3 * tmp_4
    tmp_6 = torch.sigmoid(tmp_2)
    tmp_7 = in_2 * tmp_6
    tmp_8 = torch.nn.functional.interpolate(tmp_7, (64, 64), None, 'bilinear', False)
    tmp_9 = tmp_5 + tmp_8
    return tmp_9


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


# =============================================================================
# KERNEL 1: 1×1 Convolution implemented as a tiled matrix multiplication
#
# in5:    [B, IC, HW]   where HW = H*W = 256, IC=OC=128
# weight: [OC, IC, 1,1] treated as [OC, IC]
# bias:   [OC]
# out:    [B, OC, HW]
#
# out[b, oc, hw] = Σ_ic weight[oc,ic] * in5[b,ic,hw] + bias[oc]
# =============================================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 32, 'BLOCK_IC': 32}, num_warps=4),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=4),
        triton.Config({'BLOCK_OC': 64, 'BLOCK_HW': 64, 'BLOCK_IC': 32}, num_warps=8),
        triton.Config({'BLOCK_OC': 32, 'BLOCK_HW': 32, 'BLOCK_IC': 64}, num_warps=4),
    ],
    key=['B', 'IC'],
)
@triton.jit
def conv1x1_kernel(
    in5_ptr, weight_ptr, bias_ptr, out_ptr,
    B, IC, OC, HW,
    BLOCK_OC: tl.constexpr,
    BLOCK_HW: tl.constexpr,
    BLOCK_IC: tl.constexpr,
):
    pid = tl.program_id(0)

    num_oc_tiles = (OC + BLOCK_OC - 1) // BLOCK_OC
    num_hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW

    b      = pid // (num_oc_tiles * num_hw_tiles)
    rem    = pid  % (num_oc_tiles * num_hw_tiles)
    pid_oc = rem  // num_hw_tiles
    pid_hw = rem  % num_hw_tiles

    offs_oc = pid_oc * BLOCK_OC + tl.arange(0, BLOCK_OC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    acc = tl.zeros((BLOCK_OC, BLOCK_HW), dtype=tl.float32)

    for k in range(0, IC, BLOCK_IC):
        offs_ic = k + tl.arange(0, BLOCK_IC)

        # Load weight tile [BLOCK_OC, BLOCK_IC]
        # weight[oc, ic] at weight_ptr + oc*IC + ic  (contiguous row)
        w_mask = (offs_oc[:, None] < OC) & (offs_ic[None, :] < IC)
        w_tile = tl.load(
            weight_ptr + offs_oc[:, None] * IC + offs_ic[None, :],
            mask=w_mask, other=0.0
        ).to(tl.float32)

        # Load input tile [BLOCK_IC, BLOCK_HW]
        # in5[b, ic, hw] at in5_ptr + b*IC*HW + ic*HW + hw
        a_mask = (offs_ic[:, None] < IC) & (offs_hw[None, :] < HW)
        a_tile = tl.load(
            in5_ptr + b * IC * HW + offs_ic[:, None] * HW + offs_hw[None, :],
            mask=a_mask, other=0.0
        ).to(tl.float32)

        # Accumulate [BLOCK_OC, BLOCK_IC] @ [BLOCK_IC, BLOCK_HW]
        acc += tl.dot(w_tile, a_tile)

    # Add bias broadcast over HW dimension
    bias_vals = tl.load(bias_ptr + offs_oc, mask=offs_oc < OC, other=0.0).to(tl.float32)
    acc += bias_vals[:, None]

    # Store output[b, oc, hw] at out_ptr + b*OC*HW + oc*HW + hw
    out_mask = (offs_oc[:, None] < OC) & (offs_hw[None, :] < HW)
    tl.store(
        out_ptr + b * OC * HW + offs_oc[:, None] * HW + offs_hw[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=out_mask,
    )


# =============================================================================
# KERNEL 2: Fused bilinear-upsample + sigmoid + element-mul + add
#
# For each output element (b, c, oh, ow) in [B, C, 64, 64]:
#
#   Branch A:
#     interp_in4 = bilinear_interp(in4[b,c], oh, ow)   [16×16 → 64×64]
#     branch_A   = in3[b,c,oh,ow] * sigmoid(interp_in4)
#
#   Branch B:
#     For the 4 bilinear corners (yi, xi) in the 16×16 grid:
#       val_yx = in2[b,c,yi,xi] * sigmoid(conv[b,c,yi,xi])
#     branch_B = bilinear_interp over those 4 val_yx values
#
#   output[b,c,oh,ow] = branch_A + branch_B
#
# Fixed: IH=IW=16, OH=OW=64, align_corners=False  →  scale = 0.25
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
def fused_bga_kernel(
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

    # --- decode flat index → (n, c, oh, ow) --------------------------------
    oh_ow = offsets % (OH * OW)
    nc    = offsets // (OH * OW)
    oh    = oh_ow // OW
    ow    = oh_ow % OW
    n     = nc // C
    c     = nc % C

    # --- bilinear source coordinates (align_corners=False, scale=0.25) ----
    # in_y = (oh + 0.5) * (IH/OH) - 0.5 = oh * 0.25 - 0.375
    in_y = oh.to(tl.float32) * 0.25 - 0.375
    in_x = ow.to(tl.float32) * 0.25 - 0.375

    # safe floor for potentially negative values:
    # Triton int32 cast truncates toward zero, so correct for negatives
    y_trunc = in_y.to(tl.int32)
    x_trunc = in_x.to(tl.int32)
    y_floor = tl.where(in_y < y_trunc.to(tl.float32), y_trunc - 1, y_trunc)
    x_floor = tl.where(in_x < x_trunc.to(tl.float32), x_trunc - 1, x_trunc)

    wy1 = in_y - y_floor.to(tl.float32)
    wx1 = in_x - x_floor.to(tl.float32)
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    # clamped 16×16 indices
    y0 = tl.maximum(y_floor,     0)
    y1 = tl.minimum(y_floor + 1, IH - 1)
    x0 = tl.maximum(x_floor,     0)
    x1 = tl.minimum(x_floor + 1, IW - 1)

    base_16 = (n * C + c) * (IH * IW)
    base_64 = (n * C + c) * (OH * OW)

    # --- Branch A: bilinear(in4) → sigmoid → × in3 -----------------------
    in4_00 = tl.load(in4_ptr + base_16 + y0 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in4_01 = tl.load(in4_ptr + base_16 + y0 * IW + x1, mask=mask, other=0.0).to(tl.float32)
    in4_10 = tl.load(in4_ptr + base_16 + y1 * IW + x0, mask=mask, other=0.0).to(tl.float32)
    in4_11 = tl.load(in4_ptr + base_16 + y1 * IW + x1, mask=mask, other=0.0).to(tl.float32)

    in4_blerp = (wy0 * wx0 * in4_00 + wy0 * wx1 * in4_01
               + wy1 * wx0 * in4_10 + wy1 * wx1 * in4_11)
    sig_in4 = tl.sigmoid(in4_blerp)

    in3_val  = tl.load(in3_ptr + base_64 + oh * OW + ow, mask=mask, other=0.0).to(tl.float32)
    branch_a = in3_val * sig_in4

    # --- Branch B: [sigmoid(conv) × in2] bilinearly interpolated ----------
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

    # --- Final add and store -----------------------------------------------
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
def full_bisenet_fusion(in_0, in_1, in_2, in_3, in_4, in_5):
    # in_0: bias  [OC=128]
    # in_1: weight [OC=128, IC=128, 1, 1]
    # in_2: detail features [B, 128, 16, 16]
    # in_3: context features [B, 128, 64, 64]
    # in_4: attention input [B, 128, 16, 16]
    # in_5: conv input [B, 128, 16, 16]

    B  = in_5.shape[0]
    IC = in_5.shape[1]   # 128
    OC = in_1.shape[0]   # 128
    H  = in_5.shape[2]   # 16
    W  = in_5.shape[3]   # 16
    HW = H * W           # 256

    # ------------------------------------------------------------------
    # Step 1: 1×1 convolution  →  conv_out [B, OC, H, W]
    # ------------------------------------------------------------------
    conv_out = torch.empty_like(in_5)   # same shape/dtype as input

    def grid_conv(meta):
        n_oc = (OC + meta['BLOCK_OC'] - 1) // meta['BLOCK_OC']
        n_hw = (HW + meta['BLOCK_HW'] - 1) // meta['BLOCK_HW']
        return (B * n_oc * n_hw,)

    # weight viewed as [OC, IC] — strides [IC,1,1,1] so offset = oc*IC+ic ✓
    conv1x1_kernel[grid_conv](
        in_5, in_1, in_0, conv_out,
        B, IC, OC, HW,
    )

    # ------------------------------------------------------------------
    # Step 2: Fused bilinear-upsample + sigmoid + mul + add
    #         → out [B, C, 64, 64]
    # ------------------------------------------------------------------
    out = torch.empty_like(in_3)   # [B, 128, 64, 64], same dtype as in_3

    C       = IC
    N_TOTAL = B * C * 64 * 64

    def grid_fused(meta):
        return ((N_TOTAL + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    fused_bga_kernel[grid_fused](
        conv_out, in_2, in_3, in_4, out,
        B, C,
    )

    return out


# =============================================================================
# REPLACEMENT HOOK
# =============================================================================
def replacement_func():
    return full_bisenet_fusion