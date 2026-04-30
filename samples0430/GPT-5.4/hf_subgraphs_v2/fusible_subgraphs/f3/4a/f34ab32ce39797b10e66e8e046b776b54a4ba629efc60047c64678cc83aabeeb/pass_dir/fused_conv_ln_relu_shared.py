import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv1x1_layernorm_relu_kernel(
    x_ptr,
    w_ptr,
    conv_b_ptr,
    ln_b_ptr,
    ln_w_ptr,
    out_ptr,
    n_rows,
    n_cols,
    k_dim,
    stride_xn,
    stride_xc,
    stride_wn,
    stride_wk,
    stride_outn,
    stride_outc,
    stride_conv_b,
    stride_ln_b,
    stride_ln_w,
    EPS: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return

    offs_c = tl.arange(0, BLOCK_C)
    c_mask = offs_c < n_cols
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    x_row_ptr = x_ptr + pid * stride_xn
    for k0 in tl.range(0, k_dim, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < k_dim
        x_vals = tl.load(x_row_ptr + offs_k * stride_xc, mask=k_mask, other=0.0).to(tl.float32)
        w_vals = tl.load(
            w_ptr + offs_c[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=c_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w_vals * x_vals[None, :], axis=1)

    conv_b = tl.load(conv_b_ptr + offs_c * stride_conv_b, mask=c_mask, other=0.0).to(tl.float32)
    vals = acc + conv_b

    vals_masked = tl.where(c_mask, vals, 0.0)
    mean = tl.sum(vals_masked, axis=0) / n_cols
    centered = tl.where(c_mask, vals - mean, 0.0)
    var = tl.sum(centered * centered, axis=0) / n_cols
    inv_std = tl.rsqrt(var + EPS)

    ln_w = tl.load(ln_w_ptr + offs_c * stride_ln_w, mask=c_mask, other=1.0).to(tl.float32)
    ln_b = tl.load(ln_b_ptr + offs_c * stride_ln_b, mask=c_mask, other=0.0).to(tl.float32)
    out_vals = centered * inv_std * ln_w + ln_b
    out_vals = tl.maximum(out_vals, 0.0)

    tl.store(out_ptr + pid * stride_outn + offs_c * stride_outc, out_vals, mask=c_mask)


@torch.fx.wrap
def fused_conv1x1_layernorm_relu(conv_bias, weight, ln_bias, ln_weight, x):
    n_rows = x.shape[0]
    k_dim = x.shape[1]
    n_cols = weight.shape[0]

    out = torch.empty((n_rows, n_cols, 1, 1), device=x.device, dtype=x.dtype)

    if n_cols <= 16:
        block_c = 16
        num_warps = 1
    elif n_cols <= 32:
        block_c = 32
        num_warps = 1
    elif n_cols <= 64:
        block_c = 64
        num_warps = 2
    else:
        block_c = 128
        num_warps = 4

    if k_dim <= 128:
        block_k = 32
    elif k_dim <= 384:
        block_k = 64
    else:
        block_k = 128

    grid = (n_rows,)
    fused_conv1x1_layernorm_relu_kernel[grid](
        x,
        weight,
        conv_bias,
        ln_bias,
        ln_weight,
        out,
        n_rows,
        n_cols,
        k_dim,
        x.stride(0),
        x.stride(1),
        weight.stride(0),
        weight.stride(1),
        out.stride(0),
        out.stride(1),
        conv_bias.stride(0),
        ln_bias.stride(0),
        ln_weight.stride(0),
        EPS=1e-5,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    return out