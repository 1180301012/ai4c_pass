import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return (tmp_2,)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def fused_scaled_softmax_kernel(
    x_ptr, out_ptr,
    B, H, M, N,
    stride_b, stride_h, stride_m, stride_n,
    out_stride_b, out_stride_h, out_stride_m, out_stride_n,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)  # which block of M rows
    bh = tl.program_id(1)     # combined (b, h) index

    b = bh // H
    h = bh % H

    m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offsets = tl.arange(0, BLOCK_N)

    m_mask = m_offsets < M
    n_mask = n_offsets < N

    # Load 2D block: x[b, h, m_offsets, n_offsets]
    x_ptrs = x_ptr + b * stride_b + h * stride_h + m_offsets[:, None] * stride_m + n_offsets[None, :] * stride_n
    x = tl.load(x_ptrs, mask=m_mask[:, None] & n_mask[None, :], other=0.0).to(tl.float32)

    # Scale
    x = x * scale

    # Replace masked (out-of-bounds) elements with very negative value for softmax
    # For valid m rows: valid n values are actual data, invalid n values become -1e30
    # For invalid m rows: all values become -1e30 (but these rows are masked in store)
    x = tl.where(m_mask[:, None] & n_mask[None, :], x, -1e30)

    # Softmax per row (over N dimension, axis=1)
    row_max = tl.max(x, axis=1)  # max over N for each M row
    x_exp = tl.exp(x - row_max[:, None])
    row_sum = tl.sum(x_exp, axis=1)  # sum over N for each M row
    result = x_exp / row_sum[:, None]

    # Store in [B, H, M, N] order (contiguous write, out_stride_n=1)
    out_ptrs = out_ptr + b * out_stride_b + h * out_stride_h + m_offsets[:, None] * out_stride_m + n_offsets[None, :] * out_stride_n
    tl.store(out_ptrs, result, mask=m_mask[:, None] & n_mask[None, :])


@torch.fx.wrap
def kernel_wrapper(x):
    B, H, M, N = x.shape

    # Compute scale+softmax, store in [B, H, M, N] order (contiguous writes)
    softmax_out = torch.empty(B, H, M, N, dtype=x.dtype, device=x.device)

    stride_b, stride_h, stride_m, stride_n = x.stride()
    out_stride_b, out_stride_h, out_stride_m, out_stride_n = softmax_out.stride()

    scale = 0.1767766952966369

    BLOCK_M = 4  # Process 4 rows per program for better utilization
    BLOCK_N = triton.next_power_of_2(N)

    grid = ((M + BLOCK_M - 1) // BLOCK_M, B * H)

    fused_scaled_softmax_kernel[grid](
        x_ptr=x, out_ptr=softmax_out,
        B=B, H=H, M=M, N=N,
        stride_b=stride_b, stride_h=stride_h, stride_m=stride_m, stride_n=stride_n,
        out_stride_b=out_stride_b, out_stride_h=out_stride_h, out_stride_m=out_stride_m, out_stride_n=out_stride_n,
        scale=scale,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )

    # Transpose is a free view operation (no data movement on GPU)
    return softmax_out.transpose(-2, -1)


def replacement_func():
    return kernel_wrapper