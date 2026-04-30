import torch
import triton
import triton.language as tl


def pattern(bias, weight, conv_input, cat_input):
    conv = torch.conv2d(conv_input, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    stacked = torch.stack([conv], dim=0)
    summed = stacked.sum(dim=0)
    result = torch.cat([summed, cat_input], 1)
    return result


def replacement_args(bias, weight, conv_input, cat_input):
    return (bias, weight, conv_input, cat_input)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_fused_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    M, N, K,
    stride_input_b,
    stride_output_b,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    1x1 convolution as batched GEMM writing directly to output.
    Computes: output[b, n, m] = sum_k weight[n, k] * input[b, k, m] + bias[n]
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    input_batch = input_ptr + pid_b * stride_input_b

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # Load weight[n, k]: shape [BLOCK_N, BLOCK_K]
        w_ptrs = weight_ptr + n_offs[:, None] * K + k_offs[None, :]
        w_mask = (n_offs[:, None] < N) & (k_offs[None, :] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        # Load input[k, m]: shape [BLOCK_K, BLOCK_M]
        i_ptrs = input_batch + k_offs[:, None] * M + m_offs[None, :]
        i_mask = (k_offs[:, None] < K) & (m_offs[None, :] < M)
        inp = tl.load(i_ptrs, mask=i_mask, other=0.0)

        # GEMM: [BLOCK_N, BLOCK_K] @ [BLOCK_K, BLOCK_M] -> [BLOCK_N, BLOCK_M]
        acc += tl.dot(w, inp)

    # Add bias
    bias_vals = tl.load(bias_ptr + n_offs, mask=n_offs < N, other=0.0)
    acc += bias_vals[:, None]

    # Store to output
    output_batch = output_ptr + pid_b * stride_output_b
    out_ptrs = output_batch + n_offs[:, None] * M + m_offs[None, :]
    out_mask = (n_offs[:, None] < N) & (m_offs[None, :] < M)
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.jit
def copy_cat_kernel(
    src_ptr, dst_ptr,
    n_per_batch,
    src_batch_stride,
    dst_batch_stride,
    dst_channel_offset,
    BLOCK_SIZE: tl.constexpr,
):
    """Copy cat_input to the second channel portion of the output tensor."""
    pid_block = tl.program_id(0)
    pid_batch = tl.program_id(1)

    offs = pid_block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_per_batch

    src_idx = pid_batch * src_batch_stride + offs
    dst_idx = pid_batch * dst_batch_stride + dst_channel_offset + offs

    val = tl.load(src_ptr + src_idx, mask=mask)
    tl.store(dst_ptr + dst_idx, val, mask=mask)


@torch.fx.wrap
def fused_conv1x1_cat(bias, weight, conv_input, cat_input):
    B = conv_input.shape[0]
    C_in = conv_input.shape[1]
    H = conv_input.shape[2]
    W = conv_input.shape[3]
    C_out = weight.shape[0]
    HW = H * W
    cat_C = cat_input.shape[1]
    total_C = C_out + cat_C

    output = torch.empty((B, total_C, H, W), dtype=conv_input.dtype, device=conv_input.device)

    M = HW
    N = C_out
    K = C_in
    stride_input_b = C_in * HW
    stride_output_b = total_C * HW

    grid_conv = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
        B,
    )

    conv1x1_fused_kernel[grid_conv](
        conv_input, weight, bias, output,
        M, N, K,
        stride_input_b,
        stride_output_b,
    )

    # Copy cat_input to output[:, C_out:, :, :] - launch first for potential overlap
    n_per_batch = cat_C * HW
    COPY_BLOCK = 8192
    grid_copy = (triton.cdiv(n_per_batch, COPY_BLOCK), B)

    copy_cat_kernel[grid_copy](
        cat_input, output,
        n_per_batch,
        cat_C * HW,
        total_C * HW,
        C_out * HW,
        BLOCK_SIZE=COPY_BLOCK,
    )

    return output


def replacement_func():
    return fused_conv1x1_cat