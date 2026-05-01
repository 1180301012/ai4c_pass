"""
Shared fused Triton kernels:
1. fused_pool_cat_bn_relu: max_pool2d + cat + batch_norm + relu
2. fused_cat_bn_relu: cat + batch_norm + relu (no pool, inputs already at same size)
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def _fused_pool_cat_bn_relu_kernel(
    cat_base_ptr,    # [B, C1, H, W]
    pool_input_ptr,  # [B, C2, 2H, 2W]
    mean_ptr,        # [C]
    var_ptr,         # [C]
    weight_ptr,      # [C]
    bias_ptr,        # [C]
    output_ptr,      # [B, C, H, W]
    total_elements,
    C1, C2, H, W,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    C   = C1 + C2
    W2  = W * 2
    HW  = H * W
    CHW = C * HW
    C1HW  = C1 * HW
    C2HW4 = C2 * H * W * 4

    w_idx = offsets % W
    h_idx = (offsets // W) % H
    c_idx = (offsets // HW) % C
    b_idx = offsets // CHW

    mean_v   = tl.load(mean_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)
    var_v    = tl.load(var_ptr    + c_idx, mask=mask, other=1.0).to(tl.float32)
    weight_v = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bias_v   = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    is_cat   = c_idx < C1
    not_cat  = mask & (~is_cat)

    cat_c = tl.where(is_cat, c_idx, 0)
    cat_v = tl.load(
        cat_base_ptr + b_idx * C1HW + cat_c * HW + h_idx * W + w_idx,
        mask=mask & is_cat, other=0.0,
    ).to(tl.float32)

    pool_c  = tl.where(is_cat, 0, c_idx - C1)
    base    = b_idx * C2HW4 + pool_c * HW * 4
    h2 = h_idx * 2
    w2 = w_idx * 2

    p00 = tl.load(pool_input_ptr + base +  h2      * W2 + w2,     mask=not_cat, other=0.0).to(tl.float32)
    p01 = tl.load(pool_input_ptr + base +  h2      * W2 + w2 + 1, mask=not_cat, other=0.0).to(tl.float32)
    p10 = tl.load(pool_input_ptr + base + (h2 + 1) * W2 + w2,     mask=not_cat, other=0.0).to(tl.float32)
    p11 = tl.load(pool_input_ptr + base + (h2 + 1) * W2 + w2 + 1, mask=not_cat, other=0.0).to(tl.float32)
    pool_v = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))

    x = tl.where(is_cat, cat_v, pool_v)
    x = (x - mean_v) * tl.math.rsqrt(var_v + 0.001) * weight_v + bias_v
    x = tl.maximum(x, 0.0)

    if IS_FP16:
        x_out = x.to(tl.float16)
    elif IS_BF16:
        x_out = x.to(tl.bfloat16)
    else:
        x_out = x

    tl.store(output_ptr + offsets, x_out, mask=mask)


@torch.fx.wrap
def fused_pool_cat_bn_relu(cat_base, pool_input, running_mean, running_var, weight, bias):
    B, C1, H, W = cat_base.shape
    C2 = pool_input.shape[1]
    C  = C1 + C2
    output = torch.empty(B, C, H, W, dtype=cat_base.dtype, device=cat_base.device)
    total = B * C * H * W
    IS_FP16 = (cat_base.dtype == torch.float16)
    IS_BF16 = (cat_base.dtype == torch.bfloat16)
    def grid(meta):
        return (triton.cdiv(total, meta['BLOCK_SIZE']),)
    _fused_pool_cat_bn_relu_kernel[grid](
        cat_base, pool_input, running_mean, running_var, weight, bias, output,
        total, C1, C2, H, W, IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return output


# ── Simpler kernel: cat + BN + relu (no pool, inputs already same spatial size) ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def _fused_cat_bn_relu_kernel(
    a_ptr,      # [B, C1, H, W]
    b_ptr,      # [B, C2, H, W]
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    output_ptr,
    total, C1, C2, H, W,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    C   = C1 + C2
    HW  = H * W
    CHW = C * HW
    C1HW = C1 * HW
    C2HW = C2 * HW

    w = offsets % W
    h = (offsets // W) % H
    c = (offsets // HW) % C
    b = offsets // CHW

    mean_v = tl.load(mean_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    var_v  = tl.load(var_ptr    + c, mask=mask, other=1.0).to(tl.float32)
    wt     = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bs     = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    is_a = c < C1

    a_c = tl.where(is_a, c, 0)
    x_a = tl.load(a_ptr + b * C1HW + a_c * HW + h * W + w, mask=mask & is_a,  other=0.0).to(tl.float32)

    b_c = tl.where(is_a, 0, c - C1)
    x_b = tl.load(b_ptr + b * C2HW + b_c * HW + h * W + w, mask=mask & ~is_a, other=0.0).to(tl.float32)

    x = tl.where(is_a, x_a, x_b)
    x = (x - mean_v) * tl.math.rsqrt(var_v + 0.001) * wt + bs
    x = tl.maximum(x, 0.0)

    if IS_FP16:
        x = x.to(tl.float16)
    elif IS_BF16:
        x = x.to(tl.bfloat16)

    tl.store(output_ptr + offsets, x, mask=mask)


@torch.fx.wrap
def fused_cat_bn_relu(a, b, running_mean, running_var, weight, bias):
    """cat([a, b], dim=1) + batch_norm(inference) + relu — no pool."""
    B, C1, H, W = a.shape
    C2 = b.shape[1]
    C  = C1 + C2
    out = torch.empty(B, C, H, W, dtype=a.dtype, device=a.device)
    total = B * C * H * W
    IS_FP16 = (a.dtype == torch.float16)
    IS_BF16 = (a.dtype == torch.bfloat16)
    def grid(meta):
        return (triton.cdiv(total, meta['BLOCK_SIZE']),)
    _fused_cat_bn_relu_kernel[grid](
        a, b, running_mean, running_var, weight, bias, out,
        total, C1, C2, H, W, IS_FP16=IS_FP16, IS_BF16=IS_BF16,
    )
    return out