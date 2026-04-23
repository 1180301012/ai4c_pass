import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def _cat_bn_relu_kernel(
    a_ptr,
    b_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    c_total,
    c_a,
    spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    spatial_i64 = tl.full((), spatial, tl.int64)
    c_total_i64 = tl.full((), c_total, tl.int64)
    c_a_i64 = tl.full((), c_a, tl.int64)

    offs_i64 = offs.to(tl.int64)
    inner = offs_i64 % spatial_i64
    tmp = offs_i64 // spatial_i64
    c = tmp % c_total_i64
    n = tmp // c_total_i64

    a_index = (n * c_a_i64 + c) * spatial_i64 + inner
    b_index = (n * (c_total_i64 - c_a_i64) + (c - c_a_i64)) * spatial_i64 + inner

    from_a = c < c_a_i64

    a_val = tl.load(a_ptr + a_index, mask=mask & from_a, other=0.0)
    b_val = tl.load(b_ptr + b_index, mask=mask & (~from_a), other=0.0)
    x = tl.where(from_a, a_val, b_val)

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_f32 = x.to(tl.float32)
    y = (x_f32 - mean.to(tl.float32)) * tl.rsqrt(var.to(tl.float32) + eps)
    y = y * weight.to(tl.float32) + bias.to(tl.float32)
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + offs, y.to(x.dtype), mask=mask)


@torch.fx.wrap
def erfnet_pool_interp_cat_bn_relu(in_0, in_1, in_2, in_3, in_4, in_5, out_hw):
    pooled = torch.nn.functional.max_pool2d(in_5, 2, 2, 0, 1, ceil_mode=False, return_indices=False)
    up = torch.nn.functional.interpolate(pooled, out_hw, None, 'bilinear', False)

    n, c_a, h, w = in_4.shape
    c_b = up.shape[1]
    c_total = c_a + c_b
    spatial = h * w
    n_elements = n * c_total * spatial

    out = torch.empty((n, c_total, h, w), device=in_4.device, dtype=in_4.dtype)

    def grid(meta):
        return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    _cat_bn_relu_kernel[grid](
        in_4,
        up,
        in_0,
        in_1,
        in_3,
        in_2,
        out,
        n_elements,
        c_total,
        c_a,
        spatial,
        0.001,
    )
    return out