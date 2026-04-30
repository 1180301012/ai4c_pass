import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    transposed = linear.transpose(-1, -2)
    result = in_3 * transposed
    return result


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.autotune(
    configs=[
        # Large batch configs
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        # Medium batch configs
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        # Small batch configs
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['B'],
)
@triton.jit
def fused_linear_transpose_mul_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
    B, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_in2_b, stride_in2_n, stride_in2_k,
    stride_in1_m, stride_in1_k,
    stride_in3_b, stride_in3_m, stride_in3_n,
    stride_out_b, stride_out_m, stride_out_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Computes: out[b, m, n] = in_3[b, m, n] * (sum_k(in_1[m, k] * in_2[b, n, k]) + in_0[m])
    Fuses: linear(in_2, in_1, in_0).transpose(-1, -2) * in_3
    """
    pid = tl.program_id(0)
    pid_b = tl.program_id(1)

    num_n_tiles = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n_tiles
    pid_n = pid % num_n_tiles

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers for in_1[m, k] - [BLOCK_M, BLOCK_K] (coalesced: k stride=1)
    a_ptrs = in_1_ptr + offs_m[:, None] * stride_in1_m + offs_k[None, :] * stride_in1_k
    # Pointers for in_2[b, n, k] as [BLOCK_N, BLOCK_K] (coalesced: k stride=1)
    b_nk_ptrs = in_2_ptr + pid_b * stride_in2_b + offs_n[:, None] * stride_in2_n + offs_k[None, :] * stride_in2_k

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main K loop - compiler can optimize since K is constexpr
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        k_mask = k_offs < K

        # Load A [BLOCK_M, BLOCK_K]
        a_mask = (offs_m[:, None] < M) & k_mask[None, :]
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        # Load B as [BLOCK_N, BLOCK_K] - coalesced
        b_nk_mask = (offs_n[:, None] < N) & k_mask[None, :]
        b_nk = tl.load(b_nk_ptrs, mask=b_nk_mask, other=0.0)

        # Transpose B to [BLOCK_K, BLOCK_N]
        b_kn = tl.trans(b_nk)

        # Matmul: [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc += tl.dot(a, b_kn)

        # Advance
        a_ptrs += BLOCK_K * stride_in1_k
        b_nk_ptrs += BLOCK_K * stride_in2_k

    # Add bias
    bias = tl.load(in_0_ptr + offs_m, mask=offs_m < M, other=0.0)
    acc += bias[:, None]

    # Load in_3 and multiply
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    in3_ptrs = in_3_ptr + pid_b * stride_in3_b + offs_m[:, None] * stride_in3_m + offs_n[None, :] * stride_in3_n
    in3 = tl.load(in3_ptrs, mask=out_mask, other=0.0)

    result = in3 * acc.to(in3.dtype)

    # Store
    out_ptrs = out_ptr + pid_b * stride_out_b + offs_m[:, None] * stride_out_m + offs_n[None, :] * stride_out_n
    tl.store(out_ptrs, result, mask=out_mask)


@torch.fx.wrap
def fused_linear_transpose_mul(in_0, in_1, in_2, in_3):
    """
    in_0: bias [M] (M=196)
    in_1: weight [M, K] (196, 196)
    in_2: input [B, N, K] (B, 768, 196)
    in_3: multiplier [B, M, N] (B, 196, 768)
    output: [B, M, N] (B, 196, 768)
    """
    B = in_2.shape[0]
    N = in_2.shape[1]  # 768
    K = in_2.shape[2]  # 196
    M = in_1.shape[0]  # 196

    out = torch.empty_like(in_3)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']), B)

    fused_linear_transpose_mul_kernel[grid](
        in_2, in_1, in_0, in_3, out,
        B, M, N, K,
        in_2.stride(0), in_2.stride(1), in_2.stride(2),
        in_1.stride(0), in_1.stride(1),
        in_3.stride(0), in_3.stride(1), in_3.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )

    return out


def replacement_func():
    return fused_linear_transpose_mul