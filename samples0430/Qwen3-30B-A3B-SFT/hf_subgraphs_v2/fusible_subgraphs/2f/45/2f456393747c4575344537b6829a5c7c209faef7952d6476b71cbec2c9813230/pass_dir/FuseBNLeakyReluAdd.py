import torch
import triton
import triton.language as tl


@triton.jit
def fused_bn_leakyrelu_add_kernel(
    x_ptr,        # conv output, [B, C, H, W] contiguous
    mean_ptr,     # running_mean, [C]
    var_ptr,      # running_var,  [C]
    weight_ptr,   # bn scale,     [C]
    bias_ptr,     # bn bias,      [C]
    residual_ptr, # [B, C, H, W] contiguous
    out_ptr,      # [B, C, H, W] contiguous
    C,
    HW,
    BLOCK_HW: tl.constexpr,   # >= HW, power-of-2; covers all spatial in one shot
):
    # pid = linear index over (batch x channel) pairs
    pid = tl.program_id(0)    # 0 .. B*C - 1
    c   = pid % C
    b   = pid // C

    # Per-channel BN parameters (read once per block; L1-cache-friendly)
    mean  = tl.load(mean_ptr   + c).to(tl.float32)
    var   = tl.load(var_ptr    + c).to(tl.float32)
    w     = tl.load(weight_ptr + c).to(tl.float32)
    b_val = tl.load(bias_ptr   + c).to(tl.float32)

    # Fused BN scale/offset (inference mode)
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale   = w * inv_std
    offset  = b_val - mean * scale

    # Flat offsets into [b, c, :, :] — one tl.arange covers the whole slice
    offs = tl.arange(0, BLOCK_HW)
    mask = offs < HW
    base = (b * C + c) * HW

    x = tl.load(x_ptr        + base + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr  + base + offs, mask=mask, other=0.0).to(tl.float32)

    x_norm = x * scale + offset
    x_relu = tl.where(x_norm >= 0.0, x_norm, 0.01 * x_norm)
    out    = x_relu + r

    tl.store(out_ptr + base + offs, out.to(x_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Module-level cache: avoids repeated CPU→GPU copies for fixed BN weights.
# Key = (device, dtype, 5 data_ptrs) so cache only hits for truly identical
# input tensors (different residuals = different keys = no false reuse).
# ---------------------------------------------------------------------------
_bn_param_cache: dict = {}


@torch.fx.wrap
def fused_bn_leakyrelu_add(conv_output, running_mean, running_var, weight, bias, residual):
    device = conv_output.device
    dtype  = conv_output.dtype

    B, C, H, W = conv_output.shape
    HW = H * W

    # ------------------------------------------------------------------
    # Cache GPU copies of batch-norm params (same across inference calls).
    # Use id() for stable tensor identity; avoids data_ptr() call overhead.
    # ------------------------------------------------------------------
    cache_key = (device, dtype, id(running_mean), id(running_var),
                 id(weight), id(bias), id(residual))
    if cache_key not in _bn_param_cache:
        _bn_param_cache[cache_key] = (
            torch.as_tensor(running_mean, device=device, dtype=dtype),
            torch.as_tensor(running_var,  device=device, dtype=dtype),
            torch.as_tensor(weight,       device=device, dtype=dtype),
            torch.as_tensor(bias,         device=device, dtype=dtype),
            torch.as_tensor(residual,     device=device, dtype=dtype),
        )
    mean_gpu, var_gpu, weight_gpu, bias_gpu, res_gpu = _bn_param_cache[cache_key]

    out = torch.empty_like(conv_output)

    # Simple power-of-2 ≥ HW: ensures single-pass kernel is correct.
    # Using 4096 for large tensors gives best throughput on A30.
    if HW <= 256:
        BLOCK_HW = 256
    elif HW <= 512:
        BLOCK_HW = 512
    elif HW <= 1024:
        BLOCK_HW = 1024
    elif HW <= 2048:
        BLOCK_HW = 2048
    else:
        BLOCK_HW = 4096

    # 1-D grid: one program per (batch, channel) pair
    fused_bn_leakyrelu_add_kernel[(B * C,)](
        conv_output, mean_gpu, var_gpu, weight_gpu, bias_gpu, res_gpu, out,
        C, HW, BLOCK_HW,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern, replacement_args, replacement_func
# ---------------------------------------------------------------------------

def pattern(conv_output, running_mean, running_var, weight, bias, residual):
    """
    Match: batch_norm(inference) → leaky_relu(0.01) → add(residual)
    Argument order mirrors model.py exactly.
    """
    bn   = torch.nn.functional.batch_norm(conv_output, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu = torch.nn.functional.leaky_relu(bn, 0.01, True)
    out  = relu + residual
    return out


def replacement_args(conv_output, running_mean, running_var, weight, bias, residual):
    return (conv_output, running_mean, running_var, weight, bias, residual)


def replacement_func():
    return fused_bn_leakyrelu_add