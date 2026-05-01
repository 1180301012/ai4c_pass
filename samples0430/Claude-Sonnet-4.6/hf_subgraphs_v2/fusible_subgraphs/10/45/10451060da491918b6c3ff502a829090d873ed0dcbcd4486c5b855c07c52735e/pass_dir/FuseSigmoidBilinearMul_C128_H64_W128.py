import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: sigmoid + bilinear-upsample(1x4 -> 64x128) + elementwise-mul
# ---------------------------------------------------------------------------

def pattern(sigmoid_input, in_2):
    tmp_2 = torch.sigmoid(sigmoid_input)
    tmp_3 = torch.nn.functional.interpolate(tmp_2, (64, 128), None, 'bilinear', False)
    tmp_4 = in_2 * tmp_3
    return tmp_4


def replacement_args(sigmoid_input, in_2):
    return (sigmoid_input, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused sigmoid + 1-D linear interpolation (H_IN=1) + mul
# ---------------------------------------------------------------------------

@triton.jit
def _fused_sigmoid_bilinear_mul_kernel(
    sig_input_ptr,          # [1, C, 1, W_IN]  (from conv2d output)
    in_2_ptr,               # [1, C, H_OUT, W_OUT]
    output_ptr,             # [1, C, H_OUT, W_OUT]
    IS_BF16: tl.constexpr,  # bool: True → bfloat16, False → float16
    H_OUT: tl.constexpr,    # 64
    W_OUT: tl.constexpr,    # 128  (must be power-of-2 for tl.arange)
    W_IN:  tl.constexpr,    # 4
):
    """
    Grid: (C, H_OUT).  Each program handles one (channel, output-row).
    It processes all W_OUT output columns in a single vectorised pass.
    """
    c_idx = tl.program_id(0)   # channel index
    h_idx = tl.program_id(1)   # output-height index

    # ------------------------------------------------------------------
    # Load the W_IN sigmoid-input values for this channel and apply sigmoid
    # sig_input layout: [1, C, 1, W_IN] contiguous
    # offset[0, c, 0, :] = c * W_IN
    # ------------------------------------------------------------------
    sig_base = c_idx * W_IN

    v0 = tl.sigmoid(tl.load(sig_input_ptr + sig_base + 0).to(tl.float32))
    v1 = tl.sigmoid(tl.load(sig_input_ptr + sig_base + 1).to(tl.float32))
    v2 = tl.sigmoid(tl.load(sig_input_ptr + sig_base + 2).to(tl.float32))
    v3 = tl.sigmoid(tl.load(sig_input_ptr + sig_base + 3).to(tl.float32))

    # ------------------------------------------------------------------
    # Bilinear (= 1-D linear here, since H_IN=1) with align_corners=False
    #   w_src = clamp((w_out + 0.5) * (W_IN / W_OUT) - 0.5,  0, W_IN-1)
    # ------------------------------------------------------------------
    w_out_idx = tl.arange(0, W_OUT)   # shape [W_OUT]

    scale_w = float(W_IN) / float(W_OUT)          # compile-time constant
    w_src = (w_out_idx.to(tl.float32) + 0.5) * scale_w - 0.5
    w_src_clamped = tl.maximum(0.0, tl.minimum(float(W_IN - 1), w_src))

    # Floor via truncation (safe: w_src_clamped >= 0)
    w0     = w_src_clamped.to(tl.int32)
    frac_w = w_src_clamped - w0.to(tl.float32)
    w1     = tl.where(w0 + 1 < W_IN, w0 + 1, W_IN - 1)

    # Gather sigmoid values at w0 and w1 (W_IN=4, so 3-level where)
    val_w0 = tl.where(w0 == 0, v0,
              tl.where(w0 == 1, v1,
              tl.where(w0 == 2, v2, v3)))

    val_w1 = tl.where(w1 == 0, v0,
              tl.where(w1 == 1, v1,
              tl.where(w1 == 2, v2, v3)))

    # Linear interpolation in W
    interp_val = val_w0 + frac_w * (val_w1 - val_w0)

    # ------------------------------------------------------------------
    # Load in_2 row, multiply, store
    # in_2 layout: [1, C, H_OUT, W_OUT] contiguous
    # offset[0, c, h, :] = c * H_OUT * W_OUT + h * W_OUT
    # ------------------------------------------------------------------
    out_base = c_idx * H_OUT * W_OUT + h_idx * W_OUT
    in2_vals = tl.load(in_2_ptr + out_base + w_out_idx)

    result = in2_vals.to(tl.float32) * interp_val

    if IS_BF16:
        tl.store(output_ptr + out_base + w_out_idx, result.to(tl.bfloat16))
    else:
        tl.store(output_ptr + out_base + w_out_idx, result.to(tl.float16))


# ---------------------------------------------------------------------------
# Wrapper function (must be @torch.fx.wrap so FX doesn't trace inside)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_sigmoid_bilinear_mul(sigmoid_input, in_2):
    """
    sigmoid_input : [1, C, 1, W_IN]  e.g. [1, 128, 1, 4]
    in_2          : [1, C, H_OUT, W_OUT]  e.g. [1, 128, 64, 128]
    """
    C     = in_2.shape[1]            # 128
    H_OUT = in_2.shape[2]            # 64
    W_OUT = in_2.shape[3]            # 128
    W_IN  = sigmoid_input.shape[3]   # 4

    output  = torch.empty_like(in_2)
    is_bf16 = sigmoid_input.dtype == torch.bfloat16

    grid = (C, H_OUT)   # (128, 64) = 8192 programs

    _fused_sigmoid_bilinear_mul_kernel[grid](
        sigmoid_input, in_2, output,
        IS_BF16=is_bf16,
        H_OUT=H_OUT,
        W_OUT=W_OUT,
        W_IN=W_IN,
    )

    return output


# ---------------------------------------------------------------------------
# replacement_func: zero-argument, returns the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_sigmoid_bilinear_mul