import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: conv2d(1x1) -> sigmoid -> bilinear-upsample -> elementwise-mul
# Shapes:
#   in_0 (weight) : [128, 960, 1, 1]
#   in_1 (input)  : [1,  960,  1, 4]
#   in_2          : [1,  128, 64, 128]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.sigmoid(conv2d)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel
#   Grid: (C=128, H=64)  — one program per (channel, output-row)
#   Each program processes W=128 output columns:
#     1. load 4 sigmoid values for channel c  (tiny, reused across row)
#     2. for each output column ow, compute bilinear weights
#        and blend the 4 sigmoid values
#     3. load in_2[0, c, oh, ow]  (contiguous block of W=128)
#     4. multiply and store
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_W': 128}, num_warps=4),
        triton.Config({'BLOCK_W': 128}, num_warps=8),
        triton.Config({'BLOCK_W': 128}, num_warps=16),
    ],
    key=['C', 'H', 'W'],
)
@triton.jit
def fused_conv_sigmoid_interp_mul_kernel(
    conv2d_ptr,   # [1, C, 1, W_src] — flattened: C * W_src values
    in2_ptr,      # [1, C, H, W]     — flattened: C * H * W values
    out_ptr,      # [1, C, H, W]     — flattened: C * H * W values
    C, H, W,      # runtime dims (used for grid selection only)
    BLOCK_W: tl.constexpr,
):
    # Compile-time constants for this specific graph
    C_CONST: tl.constexpr = 128
    H_CONST: tl.constexpr = 64
    W_CONST: tl.constexpr = 128
    W_SRC: tl.constexpr = 4

    c = tl.program_id(0)   # channel index  [0, C)
    oh = tl.program_id(1)  # output row     [0, H)

    # ---- Load 4 sigmoid values for this channel ----------------------------
    # conv2d[0, c, 0, iw] lives at offset  c * W_SRC + iw  (batch=1)
    iw0 = tl.arange(0, BLOCK_W) % W_SRC   # always 0,1,2,3 (repeated)
    sig_vals = tl.load(conv2d_ptr + c * W_SRC + iw0).to(tl.float32)
    # sig_vals[i] = sigmoid(conv2d[0, c, 0, i]) for i in 0..3

    # ---- Compute bilinear interpolation coordinates -----------------------
    # align_corners=False:
    #   scale_h = H_SRC / H = 1 / 64
    #   scale_w = W_SRC / W = 4 / 128 = 1 / 32
    #   src_h = (oh + 0.5) * scale_h - 0.5
    #   src_w = (ow + 0.5) * scale_w - 0.5
    #   iw0 = floor(src_w),  iw1 = iw0+1,  sw1 = src_w - floor(src_w)
    #   (ih0=ih1 always since H_SRC=1)
    ow = tl.arange(0, BLOCK_W)            # [0, 1, ..., W-1]

    src_w = (ow.to(tl.float32) + 0.5) * (W_SRC / W_CONST) - 0.5
    # Clamp to valid range before casting to int
    src_w_clamped = tl.maximum(0.0, tl.minimum(W_CONST - 1.0, src_w))

    iw0_i = tl.cast(src_w_clamped, tl.int32)          # floor(src_w)
    iw1_i = tl.minimum(W_CONST - 1, iw0_i + 1)        # iw0+1, clamped
    sw1 = src_w - iw0_i.to(tl.float32)                # fractional part
    sw1 = tl.maximum(0.0, sw1)                         # guard against tiny neg
    sw0 = 1.0 - sw1

    # ---- Load 4 sigmoid values (re-indexed for each position) --------------
    s0 = tl.load(conv2d_ptr + c * W_SRC + iw0_i).to(tl.float32)
    s1 = tl.load(conv2d_ptr + c * W_SRC + iw1_i).to(tl.float32)

    interp = sw0 * s0 + sw1 * s1   # [BLOCK_W] — bilinear blend

    # ---- Load in_2[0, c, oh, :] and multiply --------------------------------
    # in2 layout: [1, C, H, W]  strides: [C*H*W, H*W, W, 1]
    in2_base = c * (H_CONST * W_CONST) + oh * W_CONST
    in2_vals = tl.load(in2_ptr + in2_base + ow).to(tl.float32)

    result = in2_vals * interp

    # ---- Store --------------------------------------------------------------
    tl.store(out_ptr + in2_base + ow, result)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap so FX tracing doesn't go inside it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv_sigmoid_interp_mul(in_0, in_1, in_2):
    """
    in_0 : weight  [128, 960, 1, 1]
    in_1 : input   [1, 960, 1, 4]
    in_2 : scale   [1, 128, 64, 128]
    returns:       [1, 128, 64, 128]
    """
    C, H, W = 128, 64, 128
    out = torch.empty_like(in_2)

    grid = lambda meta: (C, H)

    fused_conv_sigmoid_interp_mul_kernel[grid](
        in_0,   # conv2d input / weight
        in_1,   # in_2
        out,
        C, H, W,
    )

    return out


def replacement_func():
    return fused_conv_sigmoid_interp_mul