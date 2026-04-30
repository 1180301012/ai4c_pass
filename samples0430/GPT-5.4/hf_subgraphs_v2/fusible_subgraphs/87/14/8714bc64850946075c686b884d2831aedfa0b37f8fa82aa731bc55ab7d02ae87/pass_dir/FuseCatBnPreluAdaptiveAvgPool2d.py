import torch
import triton
import triton.language as tl


C_TOTAL = 128
C_HALF = 64
EPS = 0.001
BLOCK_HW = 256


# Pattern matching function
# Matches:
#   cat([x0, x1], dim=1) -> batch_norm(eval) -> prelu
# and returns tmp_9, which is observable outside the matched subgraph.

def pattern(x0, x1, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    tmp_7 = torch.cat([x0, x1], 1)
    tmp_8 = torch.nn.functional.batch_norm(
        tmp_7, running_mean, running_var, bn_weight, bn_bias, False, 0.1, 0.001
    )
    tmp_9 = torch.prelu(tmp_8, prelu_weight)
    return tmp_9


# Argument extraction function

def replacement_args(x0, x1, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    return (x0, x1, running_mean, running_var, bn_weight, bn_bias, prelu_weight)


@triton.jit
def fused_cat_bn_prelu_kernel(
    x0_ptr,
    x1_ptr,
    running_mean_ptr,
    running_var_ptr,
    bn_weight_ptr,
    bn_bias_ptr,
    prelu_weight_ptr,
    out_ptr,
    plane,
    BLOCK: tl.constexpr,
    C_HALF_CONST: tl.constexpr,
    C_TOTAL_CONST: tl.constexpr,
):
    pid_hw = tl.program_id(0)
    c = tl.program_id(1)
    b = tl.program_id(2)

    offs = pid_hw * BLOCK + tl.arange(0, BLOCK)
    mask = offs < plane

    local_c = c - tl.where(c < C_HALF_CONST, 0, C_HALF_CONST)
    src_idx = (b * C_HALF_CONST + local_c) * plane + offs
    out_idx = (b * C_TOTAL_CONST + c) * plane + offs

    x0_vals = tl.load(x0_ptr + src_idx, mask=mask & (c < C_HALF_CONST), other=0.0)
    x1_vals = tl.load(x1_ptr + src_idx, mask=mask & (c >= C_HALF_CONST), other=0.0)
    x = x0_vals + x1_vals
    x_f32 = x.to(tl.float32)

    mean = tl.load(running_mean_ptr + c).to(tl.float32)
    var = tl.load(running_var_ptr + c).to(tl.float32)
    gamma = tl.load(bn_weight_ptr + c).to(tl.float32)
    beta = tl.load(bn_bias_ptr + c).to(tl.float32)
    alpha = tl.load(prelu_weight_ptr + c).to(tl.float32)

    inv_std = tl.rsqrt(var + 0.001)
    scale = gamma * inv_std
    shift = beta - mean * scale
    y = x_f32 * scale + shift
    y = tl.where(y >= 0, y, y * alpha)

    tl.store(out_ptr + out_idx, y.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_cat_bn_prelu(x0, x1, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    b = x0.shape[0]
    h = x0.shape[2]
    w = x0.shape[3]
    plane = h * w
    n_tiles = triton.cdiv(plane, BLOCK_HW)

    out = torch.empty((b, C_TOTAL, h, w), device=x0.device, dtype=x0.dtype)

    grid = (n_tiles, C_TOTAL, b)
    fused_cat_bn_prelu_kernel[grid](
        x0,
        x1,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        prelu_weight,
        out,
        plane,
        BLOCK=BLOCK_HW,
        C_HALF_CONST=C_HALF,
        C_TOTAL_CONST=C_TOTAL,
        num_warps=4,
        num_stages=2,
    )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_cat_bn_prelu