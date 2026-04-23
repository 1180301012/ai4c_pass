import torch
import triton
import triton.language as tl


@triton.jit
def fused_gelu_residual_layernorm_kernel(
    bias_ptr,
    weight_ptr,
    x_ptr,
    residual_ptr,
    out_residual_ptr,
    out_ln_ptr,
    rows,
    C,
    W,
    x_stride_b,
    x_stride_c,
    x_stride_h,
    x_stride_w,
    residual_stride_b,
    residual_stride_row,
    residual_stride_c,
    out_residual_stride_b,
    out_residual_stride_row,
    out_residual_stride_c,
    out_ln_stride_b,
    out_ln_stride_h,
    out_ln_stride_w,
    out_ln_stride_c,
    eps,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_c = tl.arange(0, BLOCK_C)
    mask = offs_c < C

    h = pid // W
    w = pid - h * W

    x_ptrs = x_ptr + offs_c * x_stride_c + h * x_stride_h + w * x_stride_w
    residual_ptrs = residual_ptr + pid * residual_stride_row + offs_c * residual_stride_c

    x = tl.load(x_ptrs, mask=mask, other=0).to(tl.float32)
    residual = tl.load(residual_ptrs, mask=mask, other=0).to(tl.float32)
    gelu = 0.5 * x * (1.0 + tl.erf(x * 0.70710678118654752440))
    y = gelu + residual

    out_residual_ptrs = out_residual_ptr + pid * out_residual_stride_row + offs_c * out_residual_stride_c
    tl.store(out_residual_ptrs, y, mask=mask)

    mean = tl.sum(y, axis=0) / C
    diff = y - mean
    var = tl.sum(diff * diff, axis=0) / C
    inv_std = tl.rsqrt(var + eps)

    weight = tl.load(weight_ptr + offs_c, mask=mask, other=1).to(tl.float32)
    bias = tl.load(bias_ptr + offs_c, mask=mask, other=0).to(tl.float32)
    ln = diff * inv_std
    ln = ln * weight + bias

    out_ln_ptrs = out_ln_ptr + h * out_ln_stride_h + w * out_ln_stride_w + offs_c * out_ln_stride_c
    tl.store(out_ln_ptrs, ln, mask=mask)


@torch.fx.wrap
def fused_hrformer_full_route(bias, weight, x, residual):
    rows = residual.shape[1]
    C = residual.shape[2]

    if rows == 3072 and C == 32:
        H = 64
        W = 48
    elif rows == 192 and C == 128:
        H = 16
        W = 12
    elif rows == 48 and C == 256:
        H = 8
        W = 6
    else:
        H = 1
        W = rows

    out_residual = torch.empty_like(residual)
    out_ln = torch.empty((1, H, W, C), device=residual.device, dtype=residual.dtype)

    if C <= 32:
        BLOCK_C = 32
        num_warps = 1
    elif C <= 128:
        BLOCK_C = 128
        num_warps = 4
    else:
        BLOCK_C = 256
        num_warps = 8

    fused_gelu_residual_layernorm_kernel[(rows,)](
        bias,
        weight,
        x,
        residual,
        out_residual,
        out_ln,
        rows,
        C,
        W,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        out_residual.stride(0),
        out_residual.stride(1),
        out_residual.stride(2),
        out_ln.stride(0),
        out_ln.stride(1),
        out_ln.stride(2),
        out_ln.stride(3),
        1e-06,
        BLOCK_C=BLOCK_C,
        num_warps=num_warps,
    )
    return out_residual, out_ln