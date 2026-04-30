import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_linear_reshape_softmax_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    rows,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_G: tl.constexpr,
    GROUP: tl.constexpr,
):
    offs_m = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_K)
    offs_g = tl.arange(0, BLOCK_G)

    mask_m = offs_m < rows
    mask_g = offs_g < GROUP

    x_mat = tl.load(
        x_ptr + offs_m[:, None] * BLOCK_K + offs_k[None, :],
        mask=mask_m[:, None],
        other=0.0,
    )

    w0_t = tl.load(
        w_ptr + offs_k[:, None] + offs_g[None, :] * BLOCK_K,
        mask=mask_g[None, :],
        other=0.0,
    )
    w1_t = tl.load(
        w_ptr + offs_k[:, None] + (offs_g[None, :] + GROUP) * BLOCK_K,
        mask=mask_g[None, :],
        other=0.0,
    )

    logits0 = tl.dot(x_mat, w0_t, out_dtype=tl.float32)
    logits1 = tl.dot(x_mat, w1_t, out_dtype=tl.float32)

    b0 = tl.load(b_ptr + offs_g, mask=mask_g, other=0.0).to(tl.float32)
    b1 = tl.load(b_ptr + offs_g + GROUP, mask=mask_g, other=0.0).to(tl.float32)
    logits0 = logits0 + b0[None, :]
    logits1 = logits1 + b1[None, :]

    neg_large = -1.0e20
    logits0 = tl.where(mask_m[:, None] & mask_g[None, :], logits0, neg_large)
    logits1 = tl.where(mask_m[:, None] & mask_g[None, :], logits1, neg_large)

    max0 = tl.max(logits0, axis=1)
    max1 = tl.max(logits1, axis=1)

    exp0 = tl.where(mask_m[:, None] & mask_g[None, :], tl.exp(logits0 - max0[:, None]), 0.0)
    exp1 = tl.where(mask_m[:, None] & mask_g[None, :], tl.exp(logits1 - max1[:, None]), 0.0)

    den0 = tl.sum(exp0, axis=1)
    den1 = tl.sum(exp1, axis=1)

    out0 = exp0 / den0[:, None]
    out1 = exp1 / den1[:, None]

    base = offs_m[:, None] * (GROUP * 2) + offs_g[None, :]
    store_mask = mask_m[:, None] & mask_g[None, :]
    tl.store(out_ptr + base, out0, mask=store_mask)
    tl.store(out_ptr + base + GROUP, out1, mask=store_mask)


@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    rows = x.numel() // x.shape[-1]
    out = torch.empty((rows * 2, 9, 1), device=x.device, dtype=x.dtype)

    grid = (1,)
    fused_linear_reshape_softmax_kernel[grid](
        x,
        weight,
        bias,
        out,
        rows,
        BLOCK_M=32,
        BLOCK_K=128,
        BLOCK_G=16,
        GROUP=9,
        num_warps=4,
        num_stages=1,
    )
    return out


def replacement_func():
    return fused_linear_reshape_softmax