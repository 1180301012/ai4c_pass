import torch
import triton
import triton.language as tl


@triton.jit
def fused_conv1x1_layernorm_relu_kernel(
    x_ptr,
    w_ptr,
    bias_ptr,
    gamma_ptr,
    beta_ptr,
    out_ptr,
    n_rows,
    cin,
    cout,
    stride_xn,
    stride_xc,
    stride_wo,
    stride_wi,
    stride_bias,
    stride_gamma,
    stride_beta,
    stride_on,
    stride_oc,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= n_rows:
        return

    offs_c = tl.arange(0, BLOCK_C)
    cmask = offs_c < cout

    acc = tl.load(bias_ptr + offs_c * stride_bias, mask=cmask, other=0.0).to(tl.float32)
    x_row_ptr = x_ptr + pid * stride_xn

    k = 0
    while k < cin:
        offs_k = k + tl.arange(0, BLOCK_K)
        kmask = offs_k < cin

        x = tl.load(x_row_ptr + offs_k * stride_xc, mask=kmask, other=0.0).to(tl.float32)
        w = tl.load(
            w_ptr + offs_c[:, None] * stride_wo + offs_k[None, :] * stride_wi,
            mask=cmask[:, None] & kmask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(w * x[None, :], axis=1)
        k += BLOCK_K

    acc_masked = tl.where(cmask, acc, 0.0)
    mean = tl.sum(acc_masked, axis=0) / cout
    diff = tl.where(cmask, acc - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / cout
    inv_std = tl.rsqrt(var + eps)

    gamma = tl.load(gamma_ptr + offs_c * stride_gamma, mask=cmask, other=1.0).to(tl.float32)
    beta = tl.load(beta_ptr + offs_c * stride_beta, mask=cmask, other=0.0).to(tl.float32)

    y = diff * inv_std
    y = y * gamma + beta
    y = tl.maximum(y, 0.0)

    out_ptrs = out_ptr + pid * stride_on + offs_c * stride_oc
    tl.store(out_ptrs, y, mask=cmask)


@torch.fx.wrap
def fused_conv1x1_layernorm_relu(in_0, in_1, in_2, in_3, in_4):
    n_rows = in_4.shape[0]
    cin = in_4.shape[1]
    cout = in_1.shape[0]

    out = torch.empty((n_rows, cout, 1, 1), device=in_4.device, dtype=in_4.dtype)

    if cout <= 32:
        block_c = 32
        num_warps = 2
    elif cout <= 64:
        block_c = 64
        num_warps = 4
    else:
        block_c = 128
        num_warps = 8

    block_k = 32 if cin <= 128 else 64
    grid = (n_rows,)

    fused_conv1x1_layernorm_relu_kernel[grid](
        in_4,
        in_1,
        in_0,
        in_3,
        in_2,
        out,
        n_rows,
        cin,
        cout,
        in_4.stride(0),
        in_4.stride(1),
        in_1.stride(0),
        in_1.stride(1),
        in_0.stride(0),
        in_3.stride(0),
        in_2.stride(0),
        out.stride(0),
        out.stride(1),
        1e-5,
        BLOCK_C=block_c,
        BLOCK_K=block_k,
        num_warps=num_warps,
        num_stages=2,
    )
    return out