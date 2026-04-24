import torch
import triton
import triton.language as tl


def pattern(x, running_mean, running_var, weight, bias, residual):
    bn = torch.nn.functional.batch_norm(x, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    relu = torch.nn.functional.leaky_relu(bn, 0.01, True)
    out = relu + residual
    return out


def replacement_args(x, running_mean, running_var, weight, bias, residual):
    return (x, running_mean, running_var, weight, bias, residual)


# ── Fused kernel: BN (inference) + LeakyReLU + residual Add
# 2-D grid: dim-0 = NC (batch × channel), dim-1 = spatial tile.
# base = pid_nc * HW  (vectorised to avoid per-element modulo).
# mask = hw_off < HW  handles cases where HW < BLOCK_SIZE.
@triton.jit
def _bn_leakyrelu_add_kernel(
    x_ptr,
    res_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    c_idx = pid_nc % C

    # Scalar BN params for this channel
    mean = tl.load(mean_ptr + c_idx).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx).to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    scale   = w * inv_std
    shift   = b - mean * scale

    hw_start   = pid_hw * BLOCK_SIZE
    hw_offsets = hw_start + tl.arange(0, BLOCK_SIZE)
    mask       = hw_offsets < HW

    base = pid_nc * HW   # correct for any (H,W) in NCHW layout

    x   = tl.load(x_ptr   + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr   + base + hw_offsets, mask=mask, other=0.0).to(tl.float32)

    x_bn   = x * scale + shift
    x_relu = tl.where(x_bn >= 0.0, x_bn, 0.01 * x_bn)
    out    = x_relu + res

    tl.store(out_ptr + base + hw_offsets, out.to(x_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def bn_leakyrelu_add_triton(x, running_mean, running_var, weight, bias, residual):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C

    out = torch.empty_like(x)

    # Integer ceiling division via `(-HW // BLOCK_SIZE) * -1` avoids triton.cdiv call.
    # Grid passed as a plain tuple (no lambda) to minimise Python dispatch overhead.
    num_hw = (-HW // 4096) * -1

    _bn_leakyrelu_add_kernel[(NC, num_hw)](
        x_ptr      = x,
        res_ptr    = residual,
        mean_ptr   = running_mean,
        var_ptr    = running_var,
        weight_ptr = weight,
        bias_ptr   = bias,
        out_ptr    = out,
        C          = C,
        HW         = HW,
        BLOCK_SIZE = 4096,
        num_warps  = 4,
    )

    return out


def replacement_func():
    return bn_leakyrelu_add_triton