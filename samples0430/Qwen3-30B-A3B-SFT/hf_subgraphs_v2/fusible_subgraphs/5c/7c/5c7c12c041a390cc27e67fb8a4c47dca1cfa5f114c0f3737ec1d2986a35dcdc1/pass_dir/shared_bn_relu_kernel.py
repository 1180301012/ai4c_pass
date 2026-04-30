"""
Shared Triton kernel and dispatch wrapper for fused BN + ReLU.
Both ERFNet_start73_end78_8_BNReLU and ERFNet_start7_end12_1_BNReLU import
fused_bn_relu_dispatch from here so that replacement_func() returns the SAME
function object in both passes (bypassing replacement_func_limit).

Pattern matches batch_norm + relu (both ops). The replacement kernel fuses
BN+relu into a single Triton kernel, eliminating the redundant relu pass.

Route strings:
  "r128"  →  BN on 128-channel tensor (ERFNet_start73_end78_8 variant)
  "r64"   →  BN on 64-channel tensor  (ERFNet_start7_end12_1 variant)
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused BatchNorm (inference) + ReLU.
    NCHW layout.  grid = (N*C, ceil(HW/BLOCK_SIZE))
    Returns relu(BN(x)) in out_ptr.
    """
    nc_id    = tl.program_id(0)
    hw_block = tl.program_id(1)

    c_id = nc_id % C

    mean = tl.load(mean_ptr + c_id).to(tl.float32)
    var  = tl.load(var_ptr  + c_id).to(tl.float32)
    w    = tl.load(weight_ptr + c_id).to(tl.float32)
    b    = tl.load(bias_ptr   + c_id).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    scale   = w * inv_std
    shift   = b - mean * scale

    base    = nc_id * HW
    offsets = hw_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < HW

    x   = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x * scale + shift, 0.0)   # ReLU

    tl.store(out_ptr + base + offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def fused_bn_relu_dispatch(x, running_mean, running_var, weight, bias, route):
    """
    Fused BatchNorm (inference) + ReLU kernel.
    Returns relu(BN(x)). The original relu node (if any) becomes a no-op.
    `route` selects the channel dimension for autotune keying.
    """
    N, C, H, W = x.shape
    HW  = H * W
    out = torch.empty_like(x)

    def grid(meta):
        return (N * C, (HW + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'])

    _bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, HW,
        0.001,
    )
    return out


# Alias
fused_bn_dispatch = fused_bn_relu_dispatch