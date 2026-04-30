import torch
import triton
import triton.language as tl


C_TOTAL = 128
C_HALF = 64
BLOCK_HW = 256
BLOCK_T = 16


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
    partial_ptr,
    plane,
    n_tiles,
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

    y_out = y.to(x.dtype)
    tl.store(out_ptr + out_idx, y_out, mask=mask)

    y_sum = tl.where(mask, y, 0.0)
    partial_sum = tl.sum(y_sum, axis=0)
    partial_idx = (b * C_TOTAL_CONST + c) * n_tiles + pid_hw
    tl.store(partial_ptr + partial_idx, partial_sum)


@triton.jit
def reduce_partial_gap_kernel(
    partial_ptr,
    out_ptr,
    plane,
    n_tiles,
    C_TOTAL_CONST: tl.constexpr,
    BLOCK_T_CONST: tl.constexpr,
):
    c = tl.program_id(0)
    b = tl.program_id(1)

    offs = tl.arange(0, BLOCK_T_CONST)
    mask = offs < n_tiles
    base = (b * C_TOTAL_CONST + c) * n_tiles
    vals = tl.load(partial_ptr + base + offs, mask=mask, other=0.0)
    total = tl.sum(vals, axis=0)
    mean = total / plane
    tl.store(out_ptr + b * C_TOTAL_CONST + c, mean)


@torch.fx.wrap
def fused_cat_bn_prelu_pool_view(x0, x1, running_mean, running_var, bn_weight, bn_bias, prelu_weight):
    b = x0.shape[0]
    h = x0.shape[2]
    w = x0.shape[3]
    plane = h * w
    n_tiles = triton.cdiv(plane, BLOCK_HW)

    out = torch.empty((b, C_TOTAL, h, w), device=x0.device, dtype=x0.dtype)
    pooled = torch.empty((b, C_TOTAL), device=x0.device, dtype=x0.dtype)
    partial = torch.empty((b, C_TOTAL, n_tiles), device=x0.device, dtype=torch.float32)

    grid1 = (n_tiles, C_TOTAL, b)
    fused_cat_bn_prelu_kernel[grid1](
        x0,
        x1,
        running_mean,
        running_var,
        bn_weight,
        bn_bias,
        prelu_weight,
        out,
        partial,
        plane,
        n_tiles,
        BLOCK=BLOCK_HW,
        C_HALF_CONST=C_HALF,
        C_TOTAL_CONST=C_TOTAL,
        num_warps=4,
        num_stages=2,
    )

    grid2 = (C_TOTAL, b)
    reduce_partial_gap_kernel[grid2](
        partial,
        pooled,
        plane,
        n_tiles,
        C_TOTAL_CONST=C_TOTAL,
        BLOCK_T_CONST=BLOCK_T,
        num_warps=1,
        num_stages=1,
    )

    return (out, pooled)