import torch
import triton
import triton.language as tl


@triton.jit
def _small_linear_kernel(
    x_ptr,
    w_ptr,
    out_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_om,
    stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, 512, BLOCK_K):
        k = k_start + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k[None, :] * stride_xk,
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_n[None, :] * stride_wn + k[:, None] * stride_wk,
            mask=(offs_n[None, :] < N) & (k[:, None] < K),
            other=0.0,
        )
        acc += tl.dot(x, w)

    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        acc.to(out_ptr.dtype.element_ty),
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


@triton.jit
def _gather_add_kernel(
    index_ptr,
    bias_table_ptr,
    attn_ptr,
    mask_ptr,
    out_ptr,
    B,
    H,
    BLOCK_COL: tl.constexpr,
):
    pid = tl.program_id(0)

    q = pid % 64
    t = pid // 64
    h = t % H
    b = t // H
    if b >= B:
        return

    cols = tl.arange(0, BLOCK_COL)
    col_mask = cols < 64

    idx = tl.load(index_ptr + q * 64 + cols, mask=col_mask, other=0).to(tl.int32)
    bias = tl.load(bias_table_ptr + idx * H + h, mask=col_mask, other=0.0).to(tl.float32)

    base = (((b * H + h) * 64 + q) * 64)
    attn = tl.load(attn_ptr + base + cols, mask=col_mask, other=0.0).to(tl.float32)
    mask_vals = tl.load(mask_ptr + (b * 64 + q) * 64 + cols, mask=col_mask, other=0.0).to(tl.float32)

    out = attn + bias + mask_vals + mask_vals
    tl.store(out_ptr + base + cols, out.to(out_ptr.dtype.element_ty), mask=col_mask)


@torch.fx.wrap
def relpos_bias_scores_dispatch(in_0, in_1, in_2, in_3, in_4):
    num_heads = in_1.shape[0]
    batch = in_2.shape[0]

    bias_table = torch.empty((225, num_heads), device=in_2.device, dtype=in_2.dtype)
    out = torch.empty_like(in_2)

    grid_mm = (triton.cdiv(225, 32), triton.cdiv(num_heads, 16))
    _small_linear_kernel[grid_mm](
        in_4,
        in_1,
        bias_table,
        225,
        num_heads,
        512,
        512,
        1,
        512,
        1,
        num_heads,
        1,
        BLOCK_M=32,
        BLOCK_N=16,
        BLOCK_K=64,
        num_warps=4,
        num_stages=2,
    )

    grid_add = (batch * num_heads * 64,)
    _gather_add_kernel[grid_add](
        in_0,
        bias_table,
        in_2,
        in_3,
        out,
        batch,
        num_heads,
        BLOCK_COL=64,
        num_warps=4,
        num_stages=2,
    )

    return out


def replacement_func():
    return relpos_bias_scores_dispatch