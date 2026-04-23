import torch
import triton
import triton.language as tl


def pattern(a, b, running_mean, running_var, bias, weight):
    tmp = torch.cat([a, b], 1)
    tmp = torch.nn.functional.batch_norm(tmp, running_mean, running_var, weight, bias, False, 0.1, 0.001)
    tmp = torch.nn.functional.relu(tmp, inplace=False)
    return tmp


def replacement_args(a, b, running_mean, running_var, bias, weight):
    return (a, b, running_mean, running_var, weight, bias)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['spatial'],
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
    c_total,
    c_a,
    c_b,
    spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)

    cols = col_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = cols < spatial

    c_total_i64 = tl.full((), c_total, tl.int64)
    c_a_i64 = tl.full((), c_a, tl.int64)
    c_b_i64 = tl.full((), c_b, tl.int64)
    spatial_i64 = tl.full((), spatial, tl.int64)
    row_i64 = tl.full((), row, tl.int64)

    c = row_i64 % c_total_i64
    n = row_i64 // c_total_i64
    from_a = c < c_a_i64

    cols_i64 = cols.to(tl.int64)

    a_base = (n * c_a_i64 + c) * spatial_i64
    b_base = (n * c_b_i64 + (c - c_a_i64)) * spatial_i64
    out_base = row_i64 * spatial_i64

    a_vals = tl.load(a_ptr + a_base + cols_i64, mask=mask & from_a, other=0.0)
    b_vals = tl.load(b_ptr + b_base + cols_i64, mask=mask & (~from_a), other=0.0)
    x = tl.where(from_a, a_vals, b_vals)

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    weight = tl.load(weight_ptr + c)
    bias = tl.load(bias_ptr + c)

    y = (x.to(tl.float32) - mean.to(tl.float32)) * tl.rsqrt(var.to(tl.float32) + eps)
    y = y * weight.to(tl.float32) + bias.to(tl.float32)
    y = tl.maximum(y, 0.0)

    tl.store(out_ptr + out_base + cols_i64, y.to(x.dtype), mask=mask)


@torch.fx.wrap
def fused_cat_batch_norm_relu(a, b, running_mean, running_var, weight, bias):
    n, c_a, h, w = a.shape
    c_b = b.shape[1]
    c_total = c_a + c_b
    spatial = h * w

    out = torch.empty((n, c_total, h, w), device=a.device, dtype=a.dtype)

    def grid(meta):
        return (n * c_total, triton.cdiv(spatial, meta['BLOCK_SIZE']))

    _cat_bn_relu_kernel[grid](
        a,
        b,
        running_mean,
        running_var,
        weight,
        bias,
        out,
        c_total,
        c_a,
        c_b,
        spatial,
        0.001,
    )
    return out


def replacement_func():
    return fused_cat_batch_norm_relu