import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = linear.transpose(-1, -2)
    tmp_4 = in_3 * tmp_3
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
    B, M, N, K,
    stride_in2_b, stride_in2_m, stride_in2_k,
    stride_in1_n, stride_in1_k,
    stride_in0_n,
    stride_in3_b, stride_in3_n, stride_in3_m,
    stride_out_b, stride_out_n, stride_out_m,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Program IDs
    pid_b = tl.program_id(0)
    pid = tl.program_id(1)

    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Swizzle for better L2 cache access (group M tiles)
    GROUP_M = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size = tl.minimum(GROUP_M, num_pid_m - first_pid_m)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size)
    pid_n = (pid % num_pid_in_group) // group_size

    # Offsets for the output tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Accumulator for linear result: (BLOCK_M, BLOCK_N) in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load in_2[b, m, k] -> (BLOCK_M, BLOCK_K)
        # in_2 shape: [B, M, K], row-major: stride_in2_k=1
        in_2_ptrs = in_2_ptr + pid_b * stride_in2_b + offs_m[:, None] * stride_in2_m + offs_k[None, :] * stride_in2_k
        in_2_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(in_2_ptrs, mask=in_2_mask, other=0.0)

        # Load in_1[n, k] -> (BLOCK_N, BLOCK_K)
        # in_1 shape: [N, K], row-major: stride_in1_k=1
        # Load in row-major for coalesced access, then transpose for dot
        in_1_ptrs = in_1_ptr + offs_n[:, None] * stride_in1_n + offs_k[None, :] * stride_in1_k
        in_1_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b = tl.load(in_1_ptrs, mask=in_1_mask, other=0.0)

        # acc[m, n] += in_2[m, k] * in_1[n, k] = sum_k a[m,k] * b[n,k]
        # = tl.dot(a, tl.trans(b)) since b is (N, K) and we need (K, N)
        acc += tl.dot(a, tl.trans(b))

    # Add bias: in_0[n] -> broadcast over m dimension
    bias_ptrs = in_0_ptr + offs_n * stride_in0_n
    bias_mask = offs_n < N
    bias = tl.load(bias_ptrs, mask=bias_mask, other=0.0).to(tl.float32)
    acc += bias[None, :]  # Shape: (BLOCK_M, BLOCK_N)

    # acc is linear_result[b, m, n] in (m, n) layout
    # Output[b, n, m] = in_3[b, n, m] * linear_result[b, m, n]
    # So result = in_3 * tl.trans(acc), where tl.trans(acc) is (BLOCK_N, BLOCK_M)

    # Load in_3[b, n, m] -> (BLOCK_N, BLOCK_M)
    in_3_ptrs = in_3_ptr + pid_b * stride_in3_b + offs_n[:, None] * stride_in3_n + offs_m[None, :] * stride_in3_m
    in_3_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    in_3 = tl.load(in_3_ptrs, mask=in_3_mask, other=0.0).to(tl.float32)

    # Multiply: result[n, m] = in_3[n, m] * acc_transposed[n, m]
    result = in_3 * tl.trans(acc)

    # Store output[b, n, m] -> (BLOCK_N, BLOCK_M)
    out_ptrs = out_ptr + pid_b * stride_out_b + offs_n[:, None] * stride_out_n + offs_m[None, :] * stride_out_m
    out_mask = (offs_n[:, None] < N) & (offs_m[None, :] < M)
    tl.store(out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    Fused implementation of: linear(in_2, in_1, in_0).transpose(-1, -2) * in_3
    
    in_0: bias [N]
    in_1: weight [N, K]
    in_2: input [B, M, K]
    in_3: multiply operand [B, N, M]
    
    Output: [B, N, M]
    """
    B = in_2.shape[0]
    M = in_2.shape[1]
    K = in_2.shape[2]
    N = in_1.shape[0]

    # Allocate output with same dtype and device as inputs
    out = torch.empty((B, N, M), device=in_2.device, dtype=in_2.dtype)

    # Grid function using META for autotune compatibility
    grid = lambda META: (
        B,
        triton.cdiv(N, META['BLOCK_N']) * triton.cdiv(M, META['BLOCK_M']),
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

    return (out,)


def replacement_func():
    return fused_linear_transpose_mul