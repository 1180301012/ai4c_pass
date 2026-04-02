import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────
# Pattern to match: BatchNorm(inference) → LeakyReLU → Add
# ──────────────────────────────────────────────────────────────────
def pattern(conv_out, running_mean, running_var, bn_weight, bn_bias, residual):
    tmp_6 = torch.nn.functional.batch_norm(conv_out, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 1e-05)
    tmp_7 = torch.nn.functional.leaky_relu(tmp_6, 0.01, True)
    tmp_8 = tmp_7 + residual
    return tmp_8


def replacement_args(conv_out, running_mean, running_var, bn_weight, bn_bias, residual):
    return (conv_out, running_mean, running_var, bn_weight, bn_bias, residual)


# ──────────────────────────────────────────────────────────────────
# GPU cache for pre-folded BN parameters (scale, shift).
# ──────────────────────────────────────────────────────────────────
_scale_shift_cache: dict = {}


def _get_scale_shift(running_mean, running_var, bn_weight, bn_bias, device, dtype):
    """
    Pre-fold inference BN: scale = weight/sqrt(var+eps), shift = bias-mean*scale.
    Cached on GPU in target dtype; eliminates sqrt/div from kernel hot path.
    """
    key = (
        running_mean.data_ptr(), running_var.data_ptr(),
        bn_weight.data_ptr(),    bn_bias.data_ptr(),
        str(device), str(dtype),
    )
    if key not in _scale_shift_cache:
        eps = 1e-05
        m = running_mean.to(device=device, dtype=torch.float32)
        v = running_var.to(device=device, dtype=torch.float32)
        w = bn_weight.to(device=device, dtype=torch.float32)
        b = bn_bias.to(device=device, dtype=torch.float32)
        inv_std = 1.0 / (v + eps).sqrt_()
        scale   = (w * inv_std).to(dtype)
        shift   = (b - m * w * inv_std).to(dtype)
        _scale_shift_cache[key] = (scale, shift)
    return _scale_shift_cache[key]


# ──────────────────────────────────────────────────────────────────
# Kernel A: N=1 specialization — 2-D grid (HW_chunks, C)
#   • pid_1 = channel → no per-element division
#   • flat = c*HW + hw_off (n=0 always)
#   • Writes in-place to residual (avoids torch.empty_like)
# ──────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 512},   num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_HW': 1024},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 2048},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_HW': 2048},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 2048},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_HW': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_HW': 4096},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_HW': 4096},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_HW': 8192},  num_warps=16, num_stages=2),
    ],
    key=['HW'],
)
@triton.jit
def _bn_lrelu_add_n1_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    residual_ptr,
    out_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    c      = tl.program_id(1)   # channel from grid dim — no division
    pid_hw = tl.program_id(0)   # HW chunk

    s  = tl.load(scale_ptr + c).to(tl.float32)
    sh = tl.load(shift_ptr + c).to(tl.float32)

    hw_off = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    mask   = hw_off < HW
    flat   = c * HW + hw_off   # n=0 always

    x   = tl.load(x_ptr        + flat, mask=mask, other=0.0)
    res = tl.load(residual_ptr + flat, mask=mask, other=0.0)

    x_f32   = x.to(tl.float32)
    res_f32 = res.to(tl.float32)

    x_norm  = x_f32 * s + sh
    x_lrelu = tl.where(x_norm >= 0.0, x_norm, 0.01 * x_norm)
    out     = (x_lrelu + res_f32).to(x.dtype)

    tl.store(out_ptr + flat, out, mask=mask)


# ──────────────────────────────────────────────────────────────────
# Kernel B: N > 1 — 2-D grid (NHW_chunks, C)
#   • pid_1 = channel → scale/shift as scalars per CTA
#   • Autotune key includes N to avoid cross-batch-size cache conflicts
# ──────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_NHW': 2048},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_NHW': 2048},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_NHW': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_NHW': 4096},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_NHW': 4096},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_NHW': 4096},  num_warps=16, num_stages=4),
        triton.Config({'BLOCK_NHW': 8192},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_NHW': 8192},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_NHW': 8192},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_NHW': 8192},  num_warps=16, num_stages=4),
        triton.Config({'BLOCK_NHW': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_NHW': 16384}, num_warps=16, num_stages=4),
    ],
    key=['NHW'],
)
@triton.jit
def _bn_lrelu_add_nd_kernel(
    x_ptr,
    scale_ptr,
    shift_ptr,
    residual_ptr,
    out_ptr,
    C, HW, NHW, CHW,
    BLOCK_NHW: tl.constexpr,
):
    c       = tl.program_id(1)
    pid_nhw = tl.program_id(0)

    s  = tl.load(scale_ptr + c).to(tl.float32)
    sh = tl.load(shift_ptr + c).to(tl.float32)

    nhw_off = pid_nhw * BLOCK_NHW + tl.arange(0, BLOCK_NHW)
    mask    = nhw_off < NHW

    n_idx  = nhw_off // HW
    hw_idx = nhw_off - n_idx * HW
    flat   = n_idx * CHW + c * HW + hw_idx

    x   = tl.load(x_ptr        + flat, mask=mask, other=0.0)
    res = tl.load(residual_ptr + flat, mask=mask, other=0.0)

    x_f32   = x.to(tl.float32)
    res_f32 = res.to(tl.float32)

    x_norm  = x_f32 * s + sh
    x_lrelu = tl.where(x_norm >= 0.0, x_norm, 0.01 * x_norm)
    out     = (x_lrelu + res_f32).to(x.dtype)

    tl.store(out_ptr + flat, out, mask=mask)


# ──────────────────────────────────────────────────────────────────
# Host wrapper
# ──────────────────────────────────────────────────────────────────
@torch.fx.wrap
def _bn_lrelu_add(conv_out, running_mean, running_var, bn_weight, bn_bias, residual):
    """
    Fused (pre-folded BN) + LeakyReLU + Add.

    N == 1  → Kernel A: 2-D (HW_chunks, C), in-place write, no allocation.
    N  > 1  → Kernel B: 2-D (NHW_chunks, C) with software-pipelined loads.
    """
    N, C, H, W = conv_out.shape
    HW  = H * W
    device = conv_out.device
    dtype  = conv_out.dtype

    scale, shift = _get_scale_shift(running_mean, running_var,
                                    bn_weight, bn_bias, device, dtype)

    if N == 1:
        grid = lambda meta: (triton.cdiv(HW, meta['BLOCK_HW']), C)
        _bn_lrelu_add_n1_kernel[grid](
            conv_out, scale, shift, residual, residual, HW,
        )
        return residual
    else:
        NHW = N * HW
        CHW = C * HW
        out = torch.empty_like(conv_out)
        grid = lambda meta: (triton.cdiv(NHW, meta['BLOCK_NHW']), C)
        _bn_lrelu_add_nd_kernel[grid](
            conv_out, scale, shift, residual, out,
            C, HW, NHW, CHW,
        )
        return out


def replacement_func():
    return _bn_lrelu_add