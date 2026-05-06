import torch
import triton
import triton.language as tl


@triton.jit
def _cat_bn_relu64_kernel(
    in4_ptr, in5_ptr, out_ptr,
    B, C, H_in, W_in,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    N_ELEMENTS,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_ELEMENTS

    HW = H_in * W_in
    CHW64 = C * HW   # channels per sample = C * 64 * 64

    # Decode flat output index -> (b, c, h, w)
    CHW64_inv = 1.0 / CHW64
    CHW_inv   = 1.0 / (C * HW)

    b = offsets // CHW64
    rem = offsets % CHW64
    c  = rem  // HW
    hw = rem  % HW
    h  = hw   // W_in
    w  = hw   % W_in

    # Bilinear interpolation: upsample from [H_in x W_in] to [2*H_in x 2*W_in]
    # align_corners=False: h_in = (h_out + 0.5) * scale - 0.5, scale = H_in / 2H_in = 0.5
    h_in_f = (h + 0.5) * 0.5 - 0.5
    w_in_f = (w + 0.5) * 0.5 - 0.5

    h_in0 = tl.floor(h_in_f).to(tl.int32)
    w_in0 = tl.floor(w_in_f).to(tl.int32)

    # Clamp to valid range
    h_in0 = tl.maximum(h_in0, 0)
    h_in0 = tl.minimum(h_in0, H_in - 1)
    w_in0 = tl.maximum(w_in0, 0)
    w_in0 = tl.minimum(w_in0, W_in - 1)

    h_in1 = h_in0 + 1
    w_in1 = w_in0 + 1

    h_in1 = tl.minimum(h_in1, H_in - 1)
    w_in1 = tl.minimum(w_in1, W_in - 1)

    h_in2 = tl.maximum(h_in1 + 1, 0)
    w_in2 = tl.maximum(w_in1 + 1, 0)

    dh0 = h_in_f - tl.floor(h_in_f)
    dw0 = w_in_f - tl.floor(w_in_f)

    h_in1 = tl.maximum(h_in0, 0)
    w_in1 = tl.maximum(w_in0, 0)

    fn_00 = dh0 * dw0
    fn_01 = dh0 * (1.0 - dw0)
    fn_10 = (1.0 - dh0) * dw0
    fn_11 = (1.0 - dh0) * (1.0 - dw0)

    in5_base = b * CHW64 + c * HW

    v00 = tl.load(in5_ptr + in5_base + h_in0 * W_in + w_in0, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(in5_ptr + in5_base + h_in0 * W_in + w_in1, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(in5_ptr + in5_base + h_in1 * W_in + w_in0, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(in5_ptr + in5_base + h_in1 * W_in + w_in1, mask=mask, other=0.0).to(tl.float32)

    interp = fn_00 * v00 + fn_01 * v01 + fn_10 * v10 + fn_11 * v11

    # Precompute per-channel BN constants: gamma * (x - mean) / sqrt(var + eps) + beta
    mean   = tl.load(bn_mean_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    var    = tl.load(bn_var_ptr    + c, mask=mask, other=1.0).to(tl.float32)
    gamma  = tl.load(bn_weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    beta   = tl.load(bn_bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    eps   = 1e-3
    scale = gamma * (1.0 / tl.sqrt(var + eps))
    result = (interp - mean) * scale + beta

    tl.store(out_ptr + offsets, tl.maximum(result, 0.0), mask=mask)


@torch.fx.wrap
def fused_cat_bn_relu_pool64(in4, in5, bn_mean, bn_var, bn_weight, bn_bias):
    b, c, h_in, w_in = in4.shape
    out = torch.empty_like(in4)
    N_ELEMENTS = b * c * h_in * w_in
    BLOCK_SIZE = 1024
    grid = ((N_ELEMENTS + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _cat_bn_relu64_kernel[grid](
        in4, in5, out,
        b, c, h_in, w_in,
        bn_mean, bn_var, bn_weight, bn_bias,
        N_ELEMENTS,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


# ---------------------------------------------------------------------------
# Pass interface
# ---------------------------------------------------------------------------

def pattern(in4, in5, bn_mean, bn_var, bn_weight, bn_bias):
    tmp4 = torch.nn.functional.max_pool2d(in5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    tmp5 = torch.nn.functional.interpolate(tmp4, (64, 64), None, 'bilinear', False)
    tmp6 = torch.cat([in4, tmp5], 1)
    tmp7 = torch.nn.functional.batch_norm(tmp6, bn_mean, bn_var, bn_weight, bn_bias, False, 0.1, 0.001)
    tmp8 = torch.nn.functional.relu(tmp7, inplace=False)
    return tmp8


def replacement_args(in4, in5, bn_mean, bn_var, bn_weight, bn_bias):
    return (in4, in5, bn_mean, bn_var, bn_weight, bn_bias)


def replacement_func():
    return fused_cat_bn_relu_pool64