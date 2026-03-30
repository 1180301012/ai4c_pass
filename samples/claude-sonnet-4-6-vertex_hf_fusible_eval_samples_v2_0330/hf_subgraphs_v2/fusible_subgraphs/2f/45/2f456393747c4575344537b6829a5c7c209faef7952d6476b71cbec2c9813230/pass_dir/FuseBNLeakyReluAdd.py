import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: batch_norm (inference) + leaky_relu + residual add
# ---------------------------------------------------------------------------
def pattern(conv_out, running_mean, running_var, bn_weight, bn_bias, residual):
    bn   = torch.nn.functional.batch_norm(conv_out, running_mean, running_var,
                                          bn_weight, bn_bias, False, 0.1, 1e-05)
    relu = torch.nn.functional.leaky_relu(bn, 0.01, True)
    out  = relu + residual
    return out


def replacement_args(conv_out, running_mean, running_var, bn_weight, bn_bias, residual):
    return (conv_out, running_mean, running_var, bn_weight, bn_bias, residual)


# ---------------------------------------------------------------------------
# Triton kernel: fused BN (inference) + LeakyReLU + residual add
#
# Grid layout:
#   axis-0  : pid_nc  in [0, N*C)   – one program per (batch, channel) pair
#   axis-1  : pid_hw  in [0, cdiv(H*W, BLOCK_SIZE)) – tile over spatial dim
#
# For each (n, c) we load the 4 scalar BN params once, compute scale/shift
# in fp32, then process BLOCK_SIZE spatial elements fused with leaky-relu + add.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
        triton.Config({'BLOCK_SIZE': 4096}),
    ],
    key=['HW', 'C'],
)
@triton.jit
def bn_leaky_relu_add_kernel(
    x_ptr,           # conv output  [N, C, H, W]  (NCHW, possibly fp16/bf16/fp32)
    mean_ptr,        # running_mean [C]
    var_ptr,         # running_var  [C]
    weight_ptr,      # bn gamma     [C]
    bias_ptr,        # bn beta      [C]
    residual_ptr,    # residual     [N, C, H, W]
    out_ptr,         # output       [N, C, H, W]
    HW,              # H * W
    C,               # number of channels
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)   # (n, c) index: nc = n*C + c
    pid_hw = tl.program_id(1)   # tile index over H*W

    # Channel index (used to index BN parameters)
    c = pid_nc % C

    # Load BN scalars for this channel; promote to fp32 for numerics
    mean = tl.load(mean_ptr   + c).to(tl.float32)
    var  = tl.load(var_ptr    + c).to(tl.float32)
    w    = tl.load(weight_ptr + c).to(tl.float32)
    b    = tl.load(bias_ptr   + c).to(tl.float32)

    eps     = 1e-5
    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = w * inv_std            # gamma / sqrt(var + eps)
    shift   = b - mean * scale       # beta - mean * gamma / sqrt(var + eps)

    hw_start = pid_hw * BLOCK_SIZE
    offsets  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW

    base = pid_nc * HW

    # Load in original dtype (e.g. fp16/bf16/fp32)
    x   = tl.load(x_ptr        + base + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr  + base + offsets, mask=mask, other=0.0)

    # BN in fp32
    y = x.to(tl.float32) * scale + shift

    # Leaky ReLU (negative_slope = 0.01)
    y = tl.where(y >= 0.0, y, y * 0.01)

    # Residual add in fp32
    out_f32 = y + res.to(tl.float32)

    # Cast back to input dtype and store
    tl.store(out_ptr + base + offsets, out_f32.to(x.dtype), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper (must be decorated with @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_bn_leaky_relu_add(conv_out, running_mean, running_var,
                             bn_weight, bn_bias, residual):
    N, C, H, W = conv_out.shape
    HW = H * W
    device = conv_out.device

    # BN stats / params are on CPU – move to GPU (small tensors, cheap)
    mean_d   = running_mean.to(device)
    var_d    = running_var.to(device)
    weight_d = bn_weight.to(device)
    bias_d   = bn_bias.to(device)

    out = torch.empty_like(conv_out)

    grid = lambda meta: (N * C, triton.cdiv(HW, meta['BLOCK_SIZE']))

    bn_leaky_relu_add_kernel[grid](
        conv_out, mean_d, var_d, weight_d, bias_d, residual, out,
        HW, C,
    )

    return out


# ---------------------------------------------------------------------------
# Replacement entry point (zero-argument, returns function reference)
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_bn_leaky_relu_add