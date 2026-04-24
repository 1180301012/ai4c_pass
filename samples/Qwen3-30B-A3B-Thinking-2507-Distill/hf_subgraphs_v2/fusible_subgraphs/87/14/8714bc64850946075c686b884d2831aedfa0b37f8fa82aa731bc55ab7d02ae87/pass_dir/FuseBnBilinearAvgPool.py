import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: batch_norm (inference) + prelu  (single output)
#
# tmp_9 = prelu(bn(tmp_7, ...), in_0)  is returned by the model directly,
# so it is an observable value that must appear in the pattern output.
# adaptive_avg_pool2d is NOT included here (it is cheap and already fast).
# ---------------------------------------------------------------------------
def pattern(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    bn_out = torch.nn.functional.batch_norm(x, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001)
    prelu_out = torch.prelu(bn_out, prelu_weight)
    return prelu_out


def replacement_args(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


# ---------------------------------------------------------------------------
# Triton kernel: fused BN (inference) + PReLU
#
# Grid: (N*C, ceil(HW/BLOCK_HW))
#   pid0 → (batch, channel) pair index
#   pid1 → spatial tile index
#
# Each program loads a contiguous chunk of x[n,c,:,:] (BLOCK_HW elements),
# applies BN then PReLU, and stores the result back.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def _fused_bn_prelu_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    bn_w_ptr,
    bn_b_ptr,
    prelu_slope_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid0 = tl.program_id(0)   # (batch, channel) index
    pid1 = tl.program_id(1)   # spatial tile index

    c = pid0 % C

    # BN parameters for this channel
    mean_c  = tl.load(mean_ptr      + c)
    var_c   = tl.load(var_ptr       + c)
    bn_w_c  = tl.load(bn_w_ptr      + c)
    bn_b_c  = tl.load(bn_b_ptr      + c)
    eps     = 1e-3              # eps = 0.001 from model

    prelu_c = tl.load(prelu_slope_ptr + c)

    # Spatial tile
    hw_start   = pid1 * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    base_offset = pid0 * HW
    x_offsets   = base_offset + hw_offsets

    x = tl.load(x_ptr + x_offsets, mask=hw_mask, other=0.0)

    # Upcast to fp32 for numerically stable BN + PReLU
    x_f32 = x.to(tl.float32)

    # BN inference
    inv_std = tl.rsqrt(var_c + eps)
    y = (x_f32 - mean_c) * inv_std * bn_w_c + bn_b_c

    # PReLU
    y = tl.where(y >= 0.0, y, prelu_c * y)

    # Store result (Triton auto-casts fp32 → output dtype)
    tl.store(out_ptr + x_offsets, y, mask=hw_mask)


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_prelu(x, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    N, C, H, W = x.shape
    HW = H * W

    out = torch.empty_like(x)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_HW']))

    _fused_bn_prelu_kernel[grid](
        x,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        prelu_weight,
        out,
        C,
        HW,
    )

    return out


def replacement_func():
    return fused_bn_prelu