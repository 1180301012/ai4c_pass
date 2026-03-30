"""
Pass: FuseConvBNAdd_start96
Pattern: BN(conv_out, mean, var, weight, bias) → bn_out += residual
Fuses inference BatchNorm + residual-add into one Triton kernel.
Matches start96_end99_0 and start116_end119_1 (deeppose) where bn_out += residual.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def bn_add_kernel_bn_iadd(
    conv_out_ptr, residual_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    nc_idx   = tl.program_id(0)
    hw_block = tl.program_id(1)

    c = nc_idx % C

    # Load per-channel BN params, compute in fp32 for accuracy
    mean_v  = tl.load(mean_ptr   + c).to(tl.float32)
    var_v   = tl.load(var_ptr    + c).to(tl.float32)
    wt      = tl.load(weight_ptr + c).to(tl.float32)
    bias_v  = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var_v + eps)
    scale   = wt * inv_std
    shift   = bias_v - mean_v * scale

    hw_start = hw_block * BLOCK_SIZE
    offsets  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW
    base     = nc_idx * HW

    x_raw = tl.load(conv_out_ptr + base + offsets, mask=mask, other=0.0)
    r_raw = tl.load(residual_ptr + base + offsets, mask=mask, other=0.0)

    out_f32 = x_raw.to(tl.float32) * scale + shift + r_raw.to(tl.float32)

    tl.store(output_ptr + base + offsets, out_f32.to(x_raw.dtype), mask=mask)


@torch.fx.wrap
def fused_bn_add_bn_iadd(mean, var, bias, weight, residual, conv_out):
    N, C, H, W = conv_out.shape
    HW     = H * W
    output = torch.empty_like(conv_out)
    grid   = (N * C, triton.cdiv(HW, 1))
    bn_add_kernel_bn_iadd[grid](
        conv_out, residual, output,
        mean, var, weight, bias,
        C, HW,
        eps=1e-5,
    )
    return output


def pattern(mean, var, bias, weight, residual, conv_out):
    bn_out  = torch.nn.functional.batch_norm(conv_out, mean, var, weight, bias, False, 0.1, 1e-05)
    bn_out += residual
    return (bn_out,)


def replacement_args(mean, var, bias, weight, residual, conv_out):
    return (mean, var, bias, weight, residual, conv_out)


def replacement_func():
    return fused_bn_add_bn_iadd