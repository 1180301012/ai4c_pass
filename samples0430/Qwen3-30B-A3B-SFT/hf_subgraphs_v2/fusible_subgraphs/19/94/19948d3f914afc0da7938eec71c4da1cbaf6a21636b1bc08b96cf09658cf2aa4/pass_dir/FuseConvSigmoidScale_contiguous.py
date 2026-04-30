import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    conv2d = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 4)
    tmp_3 = torch.sigmoid(conv2d)
    tmp_4 = tmp_3.view(1, -1, 1, 1)
    tmp_5 = in_2 * tmp_4
    tmp_6 = tmp_5.contiguous()
    return tmp_6


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# -----------------------------------------------------------------------
# 2-D kernel: dim-0 = channel tile (BLOCK_C=8 channels), dim-1 = spatial tile
#
# BLOCK_HW fixed at 1024 (no autotuner overhead):
#   HW=1024   → (12,  1) =  12 programs  – perfect fit, no masking
#   HW=2304   → (12,  3) =  36 programs – perfect fit
#   HW=3136   → (12,  4) =  48 programs  – last tile has 64/1024 active
#   HW=9216   → (12,  9) = 108 programs – perfect fit
#   HW=9600   → (12, 10) = 120 programs – last tile has 64/1024 active
#   HW=25600  → (12, 25) = 300 programs – perfect fit
# -----------------------------------------------------------------------
@triton.jit
def fused_conv_sigmoid_scale_kernel(
    in2_ptr,     # [1, 96, H, W]  feature map
    in3_ptr,     # [1, 32, 1, 1]  gating input
    wt_ptr,      # [96, 8, 1, 1]  weight
    bias_ptr,    # [96]           bias
    out_ptr,     # [1, 96, H, W]  output
    HW,          # H * W  (runtime scalar)
    BLOCK_C:  tl.constexpr,   # = 8  (power-of-2 channels per tile)
    BLOCK_HW: tl.constexpr,   # = 1024 (spatial elements per tile, fixed)
):
    chan_tile_id = tl.program_id(0)   # [0 .. 11]
    sp_tile_id   = tl.program_id(1)   # [0 .. ceil(HW/BLOCK_HW)-1]

    c_start = chan_tile_id * BLOCK_C
    c_offs  = c_start + tl.arange(0, BLOCK_C)   # [BLOCK_C]
    c_mask  = c_offs < 96

    hw_start = sp_tile_id * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)  # [BLOCK_HW]
    hw_mask  = hw_offs < HW

    # --- Gate scale: sigmoid(dot(wt[c,:], in3[g*8:]) + bias[c]) ---
    ic_idx   = tl.arange(0, 8)
    g_vec    = c_offs // 24
    in3_base = g_vec * 8
    in3_ptr1 = in3_ptr + in3_base[:, None] + ic_idx[None, :]   # [BLOCK_C, 8]
    in3_val  = tl.load(in3_ptr1, mask=c_mask[:, None], other=0.0).to(tl.float32)

    wt_ptr1  = wt_ptr + c_offs[:, None] * 8 + ic_idx[None, :]  # [BLOCK_C, 8]
    wt_val   = tl.load(wt_ptr1, mask=c_mask[:, None], other=0.0).to(tl.float32)

    bias_val = tl.load(bias_ptr + c_offs, mask=c_mask, other=0.0).to(tl.float32)
    scale    = tl.sigmoid(tl.sum(wt_val * in3_val, axis=1) + bias_val)  # [BLOCK_C]

    # --- 2-D broadcast-multiply ---
    ptrs   = in2_ptr + c_offs[:, None] * HW + hw_offs[None, :]
    mask2d = c_mask[:, None] & hw_mask[None, :]
    x      = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
    result = (x * scale[:, None]).to(x.dtype)
    tl.store(out_ptr + c_offs[:, None] * HW + hw_offs[None, :], result, mask=mask2d)


@torch.fx.wrap
def fused_conv_sigmoid_scale(in_0, in_1, in_2, in_3):
    """
    in_0 : bias   [96]
    in_1 : weight [96, 8, 1, 1]
    in_2 : feat   [1, 96, H, W]
    in_3 : gap    [1, 32, 1, 1]
    """
    H, W = in_2.shape[2], in_2.shape[3]
    HW   = H * W
    out  = torch.empty_like(in_2)

    BLOCK_HW = 1024
    n_tiles  = (HW + BLOCK_HW - 1) // BLOCK_HW

    fused_conv_sigmoid_scale_kernel[( 96 // 8, n_tiles)](
        in_2, in_3, in_1, in_0, out,
        HW,
        BLOCK_C=8,
        BLOCK_HW=BLOCK_HW,
        num_warps=4,
    )
    return out


def replacement_func():
    return fused_conv_sigmoid_scale