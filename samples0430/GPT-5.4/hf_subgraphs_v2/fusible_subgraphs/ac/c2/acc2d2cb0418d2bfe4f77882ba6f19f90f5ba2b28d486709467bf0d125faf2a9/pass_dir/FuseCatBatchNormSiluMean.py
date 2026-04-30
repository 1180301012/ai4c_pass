import torch
import triton
import triton.language as tl


# Pattern matching function
# Matches: cat -> batch_norm(inference) -> silu(inplace=True) -> mean((2, 3), keepdim=True)
def pattern(x0, x1, x2, x3, running_mean, running_var, bias, weight):
    tmp_8 = torch.cat([x0, x1, x2, x3], 1)
    tmp_9 = torch.nn.functional.batch_norm(tmp_8, running_mean, running_var, weight, bias, False, 0.1, 1e-05)
    tmp_10 = torch.nn.functional.silu(tmp_9, inplace=True)
    tmp_11 = tmp_10.mean((2, 3), keepdim=True)
    return (tmp_10, tmp_11)


# Argument extraction function
def replacement_args(x0, x1, x2, x3, running_mean, running_var, bias, weight):
    return (x0, x1, x2, x3, running_mean, running_var, bias, weight)


@triton.jit
def _fused_chunk_bn_silu_mean_kernel(
    x_ptr,
    running_mean_ptr,
    running_var_ptr,
    bias_ptr,
    weight_ptr,
    out_ptr,
    mean_ptr,
    N,
    C,
    H,
    W,
    OUT_C_TOTAL,
    x_stride_n,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    out_stride_n,
    out_stride_c,
    out_stride_h,
    out_stride_w,
    mean_stride_n,
    mean_stride_c,
    c_base,
    eps,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)

    n = pid // C
    c = pid % C
    out_c = c + c_base
    HW = H * W

    rm = tl.load(running_mean_ptr + out_c).to(tl.float32)
    rv = tl.load(running_var_ptr + out_c).to(tl.float32)
    b = tl.load(bias_ptr + out_c).to(tl.float32)
    w = tl.load(weight_ptr + out_c).to(tl.float32)
    inv_std = tl.rsqrt(rv + eps)

    acc = 0.0

    for hw_start in tl.range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW

        h = offs // W
        ww = offs % W

        x_offs = n * x_stride_n + c * x_stride_c + h * x_stride_h + ww * x_stride_w
        x = tl.load(x_ptr + x_offs, mask=mask, other=0).to(tl.float32)

        y = (x - rm) * inv_std * w + b
        sig = 1.0 / (1.0 + tl.exp(-y))
        z = y * sig

        out_offs = n * out_stride_n + out_c * out_stride_c + h * out_stride_h + ww * out_stride_w
        tl.store(out_ptr + out_offs, z, mask=mask)

        acc += tl.sum(tl.where(mask, z, 0.0), axis=0)

    mean_val = acc / HW
    mean_off = n * mean_stride_n + out_c * mean_stride_c
    tl.store(mean_ptr + mean_off, mean_val)


@torch.fx.wrap
def fused_cat_bn_silu_mean(x0, x1, x2, x3, running_mean, running_var, bias, weight):
    # Shapes
    n0, c0, h0, w0 = x0.shape
    n1, c1, h1, w1 = x1.shape
    n2, c2, h2, w2 = x2.shape
    n3, c3, h3, w3 = x3.shape

    assert n0 == n1 == n2 == n3
    assert h0 == h1 == h2 == h3
    assert w0 == w1 == w2 == w3

    N = n0
    H = h0
    W = w0
    C_total = c0 + c1 + c2 + c3

    out = torch.empty((N, C_total, H, W), device=x0.device, dtype=x0.dtype)
    mean = torch.empty((N, C_total, 1, 1), device=x0.device, dtype=x0.dtype)

    grid0 = (N * c0,)
    grid1 = (N * c1,)
    grid2 = (N * c2,)
    grid3 = (N * c3,)

    _fused_chunk_bn_silu_mean_kernel[grid0](
        x0,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        mean,
        N,
        c0,
        H,
        W,
        C_total,
        x0.stride(0),
        x0.stride(1),
        x0.stride(2),
        x0.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mean.stride(0),
        mean.stride(1),
        0,
        1e-05,
        BLOCK_HW=256,
    )

    _fused_chunk_bn_silu_mean_kernel[grid1](
        x1,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        mean,
        N,
        c1,
        H,
        W,
        C_total,
        x1.stride(0),
        x1.stride(1),
        x1.stride(2),
        x1.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mean.stride(0),
        mean.stride(1),
        c0,
        1e-05,
        BLOCK_HW=256,
    )

    _fused_chunk_bn_silu_mean_kernel[grid2](
        x2,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        mean,
        N,
        c2,
        H,
        W,
        C_total,
        x2.stride(0),
        x2.stride(1),
        x2.stride(2),
        x2.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mean.stride(0),
        mean.stride(1),
        c0 + c1,
        1e-05,
        BLOCK_HW=256,
    )

    _fused_chunk_bn_silu_mean_kernel[grid3](
        x3,
        running_mean,
        running_var,
        bias,
        weight,
        out,
        mean,
        N,
        c3,
        H,
        W,
        C_total,
        x3.stride(0),
        x3.stride(1),
        x3.stride(2),
        x3.stride(3),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        mean.stride(0),
        mean.stride(1),
        c0 + c1 + c2,
        1e-05,
        BLOCK_HW=256,
    )

    return (out, mean)


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return fused_cat_bn_silu_mean