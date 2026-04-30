import torch
import triton
import triton.language as tl


def pattern(in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_2, in_1)
    return linear


def replacement_args(in_1, in_2, in_3):
    return (in_1, in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    OUTPUT_DTYPE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak
    b_ptrs = b_ptr + tl.arange(0, BLOCK_K)[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    m_mask = offs_m < M
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=m_mask[:, None], other=0.0)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n)
    acc = acc + bias[None, :].to(tl.float32)

    # Store with dtype conversion
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    if OUTPUT_DTYPE == 1:
        tl.store(c_ptrs, acc.to(tl.float16), mask=m_mask[:, None])
    elif OUTPUT_DTYPE == 2:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=m_mask[:, None])
    else:
        tl.store(c_ptrs, acc, mask=m_mask[:, None])


@torch.fx.wrap
def triton_linear(in_1, in_2, in_3):
    B = in_3.shape[0]
    seq_len = in_3.shape[1]
    M = B * seq_len
    N = in_2.shape[0]
    K = in_2.shape[1]

    if in_3.dtype == torch.float16:
        output_dtype = 1
    elif in_3.dtype == torch.bfloat16:
        output_dtype = 2
    else:
        output_dtype = 0

    linear_out = torch.empty((B, seq_len, N), device=in_3.device, dtype=in_3.dtype)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    matmul_bias_kernel[grid](
        in_3, in_2, in_1, linear_out,
        M, N, K,
        K, 1,
        K, 1,
        N, 1,
        OUTPUT_DTYPE=output_dtype,
    )

    return linear_out


def replacement_func():
    return triton_linear