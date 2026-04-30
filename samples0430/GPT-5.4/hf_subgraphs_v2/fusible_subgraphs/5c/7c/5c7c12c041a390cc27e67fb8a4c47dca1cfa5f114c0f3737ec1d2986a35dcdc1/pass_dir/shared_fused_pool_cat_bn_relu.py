import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _fused_pool_cat_bn_relu_kernel(
    pool_src_ptr,
    cat_other_ptr,
    mean_ptr,
    var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    n_elements,
    cat_c,
    total_c,
    out_h,
    out_w,
    in_h,
    in_w,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    hw = out_h * out_w
    chw = total_c * hw
    nhw = cat_c * hw

    n = offs // chw
    rem = offs - n * chw
    c = rem // hw
    rem2 = rem - c * hw
    h = rem2 // out_w
    w = rem2 - h * out_w

    mean = tl.load(mean_ptr + c, mask=mask, other=0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1).to(tl.float32)

    cat_mask = mask & (c < cat_c)
    pool_mask = mask & (c >= cat_c)

    cat_offs = n * nhw + c * hw + h * out_w + w
    x_cat = tl.load(cat_other_ptr + cat_offs, mask=cat_mask, other=0).to(tl.float32)

    pc = c - cat_c
    pool_c = total_c - cat_c
    pool_chw = in_h * in_w
    pool_offs_base = n * (pool_c * pool_chw) + pc * pool_chw + (h * 2) * in_w + (w * 2)
    p00 = tl.load(pool_src_ptr + pool_offs_base, mask=pool_mask, other=float('-inf')).to(tl.float32)
    p01 = tl.load(pool_src_ptr + pool_offs_base + 1, mask=pool_mask, other=float('-inf')).to(tl.float32)
    p10 = tl.load(pool_src_ptr + pool_offs_base + in_w, mask=pool_mask, other=float('-inf')).to(tl.float32)
    p11 = tl.load(pool_src_ptr + pool_offs_base + in_w + 1, mask=pool_mask, other=float('-inf')).to(tl.float32)
    x_pool = tl.maximum(tl.maximum(p00, p01), tl.maximum(p10, p11))

    x = tl.where(c < cat_c, x_cat, x_pool)
    y = (x - mean) * tl.math.rsqrt(var + eps) * weight + bias
    y = tl.maximum(y, 0.0)
    tl.store(out_ptr + offs, y, mask=mask)


@torch.fx.wrap
def fused_pool_cat_bn_relu_dispatch(pool_src, cat_other, running_mean, running_var, bias, weight, route):
    if route == 'size256' or route == 'size128' or route == 'size64':
        n = cat_other.shape[0]
        cat_c = cat_other.shape[1]
        out_h = cat_other.shape[2]
        out_w = cat_other.shape[3]
        total_c = running_mean.shape[0]
        in_h = pool_src.shape[2]
        in_w = pool_src.shape[3]
        out = torch.empty((n, total_c, out_h, out_w), device=cat_other.device, dtype=cat_other.dtype)
        n_elements = out.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        _fused_pool_cat_bn_relu_kernel[grid](
            pool_src,
            cat_other,
            running_mean,
            running_var,
            bias,
            weight,
            out,
            n_elements,
            cat_c,
            total_c,
            out_h,
            out_w,
            in_h,
            in_w,
            0.001,
        )
        return (out,)
    return (torch.empty((0,), device=cat_other.device, dtype=cat_other.dtype),)