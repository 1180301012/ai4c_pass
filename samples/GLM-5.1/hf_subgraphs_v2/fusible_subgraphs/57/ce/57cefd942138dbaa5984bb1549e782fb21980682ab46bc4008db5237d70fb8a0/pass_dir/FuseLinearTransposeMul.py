import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return tmp_4


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 16, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_M': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_N': 128, 'BLOCK_M': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
    ],
    key=['B', 'N', 'M', 'K'],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
    B, M, N, K,
    stride_in2_b, stride_in2_m, stride_in2_k,
    stride_in1_n, stride_in1_k,
    stride_in0,
    stride_in3_b, stride_in3_n, stride_in3_m,
    stride_out_b, stride_out_n, stride_out_m,
    BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs - flatten batch and tile into 1D grid for max parallelism
    pid = tl.program_id(0)
    num_n_tiles = tl.cdiv(N, BLOCK_N)
    num_m_tiles = tl.cdiv(M, BLOCK_M)
    num_tiles_per_batch = num_n_tiles * num_m_tiles
    pid_b = pid // num_tiles_per_batch
    pid_tile = pid % num_tiles_per_batch
    pid_n = pid_tile // num_m_tiles
    pid_m = pid_tile % num_m_tiles

    # Offsets for this tile
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_n = offs_n < N
    mask_m = offs_m < M

    # Accumulate dot product in fp32
    # output[b,n,m] = in_3[b,n,m] * (sum_k in_2[b,m,k]*in_1[n,k] + in_0[n])
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load in_1[n, k] as (BLOCK_N, BLOCK_K) - weight matrix rows
        # in_1 has shape [N, K], strides (stride_in1_n, stride_in1_k)
        # K dimension is contiguous (stride_in1_k = 1 for row-major)
        in_1_ptrs = in_1_ptr + offs_n[:, None] * stride_in1_n + offs_k[None, :] * stride_in1_k
        in_1_mask = mask_n[:, None] & mask_k[None, :]
        in_1_vals = tl.load(in_1_ptrs, mask=in_1_mask, other=0.0)

        # Load in_2[b, m, k] as (BLOCK_M, BLOCK_K) - contiguous rows for better memory access
        # in_2 has shape [B, M, K], strides (stride_in2_b, stride_in2_m, stride_in2_k)
        # K dimension is contiguous (stride_in2_k = 1 for row-major)
        in_2_ptrs = in_2_ptr + pid_b * stride_in2_b + offs_m[:, None] * stride_in2_m + offs_k[None, :] * stride_in2_k
        in_2_mask = mask_m[:, None] & mask_k[None, :]
        in_2_vals = tl.load(in_2_ptrs, mask=in_2_mask, other=0.0)

        # tl.dot computes: result[n,m] = sum_k(in_1[n,k] * in_2[m,k])
        # in_1_vals: (BLOCK_N, BLOCK_K), tl.trans(in_2_vals): (BLOCK_K, BLOCK_M)
        # Result: (BLOCK_N, BLOCK_M) - already in the transposed order we need
        acc += tl.dot(in_1_vals, tl.trans(in_2_vals), allow_tf32=False)

    # Add bias in_0[n], broadcast over m dimension
    bias_ptrs = in_0_ptr + offs_n * stride_in0
    bias_vals = tl.load(bias_ptrs, mask=mask_n, other=0.0)
    acc += bias_vals[:, None]

    # Multiply by in_3[b, n, m]
    in_3_ptrs = in_3_ptr + pid_b * stride_in3_b + offs_n[:, None] * stride_in3_n + offs_m[None, :] * stride_in3_m
    in_3_mask = mask_n[:, None] & mask_m[None, :]
    in_3_vals = tl.load(in_3_ptrs, mask=in_3_mask, other=0.0)

    result = acc * in_3_vals

    # Store output[b, n, m]
    out_ptrs = out_ptr + pid_b * stride_out_b + offs_n[:, None] * stride_out_n + offs_m[None, :] * stride_out_m
    out_mask = mask_n[:, None] & mask_m[None, :]
    tl.store(out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def kernel_wrapper(in_0, in_1, in_2, in_3):
    # in_2: [B, M, K] - input to linear
    # in_1: [N, K] - weight of linear
    # in_0: [N] - bias of linear
    # in_3: [B, N, M] - multiplier
    # output: [B, N, M]

    B = in_2.shape[0]
    M = in_2.shape[1]
    K = in_2.shape[2]
    N = in_1.shape[0]

    out = torch.empty((B, N, M), dtype=in_3.dtype, device=in_3.device)

    # 1D grid: flatten batch + n_tiles + m_tiles for maximum parallelism
    grid = lambda META: (
        B * triton.cdiv(N, META['BLOCK_N']) * triton.cdiv(M, META['BLOCK_M']),
    )

    fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, N, K,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        in_1.stride(0), in_1.stride(1),
        in_0.stride(0),
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )

    return out


def replacement_func():
    return kernel_wrapper