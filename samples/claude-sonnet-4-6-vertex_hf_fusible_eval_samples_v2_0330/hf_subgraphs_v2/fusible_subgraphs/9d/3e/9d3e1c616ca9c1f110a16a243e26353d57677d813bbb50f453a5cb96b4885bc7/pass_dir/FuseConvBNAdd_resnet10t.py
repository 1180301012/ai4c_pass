"""
Pass: FuseConvBNAdd_resnet10t
Pattern: conv2d(in_5, in_0) → BN(mean=in_1, var=in_2, weight=in_4, bias=in_3) → in_6 += bn_out
Used in: resnet10t.c3_in1k_start23_end26_7
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 32},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def fused_bn_add_kernel_r10t(
    conv_out_ptr, residual_ptr, output_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    nc_idx   = tl.program_id(0)
    hw_block = tl.program_id(1)

    c = nc_idx % C

    # Load per-channel BN parameters (compute in fp32 for accuracy)
    mean   = tl.load(mean_ptr   + c).to(tl.float32)
    var    = tl.load(var_ptr    + c).to(tl.float32)
    weight = tl.load(weight_ptr + c).to(tl.float32)
    bias_v = tl.load(bias_ptr   + c).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = weight * inv_std
    shift   = bias_v - mean * scale

    hw_start = hw_block * BLOCK_SIZE
    offsets  = hw_start + tl.arange(0, BLOCK_SIZE)
    mask     = offsets < HW

    base     = nc_idx * HW
    conv_raw = tl.load(conv_out_ptr + base + offsets, mask=mask, other=0.0)
    res_raw  = tl.load(residual_ptr + base + offsets, mask=mask, other=0.0)

    out_f32 = conv_raw.to(tl.float32) * scale + shift + res_raw.to(tl.float32)

    tl.store(output_ptr + base + offsets, out_f32.to(conv_raw.dtype), mask=mask)


@torch.fx.wrap
def fused_conv_bn_add_r10t(conv_weight, run_mean, run_var, bn_bias, bn_weight,
                            conv_input, residual):
    # conv2d: PyTorch's cuDNN-optimised 1x1 conv
    conv_out = torch.conv2d(conv_input, conv_weight, None, (1, 1), (0, 0), (1, 1), 1)

    N, C, H, W = conv_out.shape
    HW     = H * W
    output = torch.empty_like(conv_out)

    grid = (N * C, triton.cdiv(HW, 1))
    fused_bn_add_kernel_r10t[grid](
        conv_out, residual, output,
        run_mean, run_var, bn_weight, bn_bias,
        C, HW,
        eps=1e-5,
    )
    return output


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    conv2d = torch.conv2d(in_5, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_6  = torch.nn.functional.batch_norm(conv2d, in_1, in_2, in_4, in_3, False, 0.1, 1e-05)
    in_6  += tmp_6
    return (in_6,)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6):
    return (in_0, in_1, in_2, in_3, in_4, in_5, in_6)


def replacement_func():
    return fused_conv_bn_add_r10t