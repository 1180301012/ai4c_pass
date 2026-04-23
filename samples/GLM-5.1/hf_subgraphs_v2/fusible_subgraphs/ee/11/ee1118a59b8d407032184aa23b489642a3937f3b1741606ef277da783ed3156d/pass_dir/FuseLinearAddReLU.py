import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_3, in_1, in_0)
    tmp_3 = in_2 + linear
    tmp_4 = tmp_3.relu_()
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


# Fused Linear + Add + ReLU kernel
# Uses L2 cache grouping and pipelining for maximum performance

@triton.autotune(
    configs=[
        # Full-N coverage configs (BLOCK_N=128 covers entire output column dimension)
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=4),
        # Full-K coverage configs (BLOCK_K=128 covers entire inner dimension, single iteration)
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=2),
        # Tiled configs for larger N
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_add_relu_kernel(
    input_ptr, weight_ptr, bias_ptr, residual_ptr, output_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_bias,
    stride_rm, stride_rn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    DTYPE: tl.constexpr,
):
    # L2 cache grouping
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
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Matmul loop
    for k_start in range(0, K, BLOCK_K):
        offs_k_curr = k_start + offs_k

        a_ptrs = input_ptr + offs_m[:, None] * stride_am + offs_k_curr[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (offs_k_curr[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = weight_ptr + offs_k_curr[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (offs_k_curr[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=True)

    # Fused: bias + residual + ReLU
    bias_ptrs = bias_ptr + offs_n * stride_bias
    bias = tl.load(bias_ptrs, mask=offs_n < N, other=0.0)
    accumulator += bias[None, :]

    r_ptrs = residual_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn
    r_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    residual = tl.load(r_ptrs, mask=r_mask, other=0.0)
    accumulator += residual

    result = tl.maximum(accumulator, 0.0).to(DTYPE)

    c_ptrs = output_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, result, mask=c_mask)


@torch.fx.wrap
def fused_linear_add_relu(bias, weight, residual, input):
    # Move CPU tensors to CUDA if needed
    if bias.device.type == 'cpu':
        bias = bias.to(input.device)
    if weight.device.type == 'cpu':
        weight = weight.to(input.device)

    M, K = input.shape
    N = weight.shape[0]  # weight is [N, K] for F.linear

    # Allocate output
    output = torch.empty((M, N), dtype=input.dtype, device=input.device)

    dtype_map = {
        torch.bfloat16: tl.bfloat16,
        torch.float16: tl.float16,
        torch.float32: tl.float32,
    }
    DTYPE = dtype_map[input.dtype]

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    fused_linear_add_relu_kernel[grid](
        input_ptr=input, weight_ptr=weight, bias_ptr=bias, residual_ptr=residual, output_ptr=output,
        M=M, N=N, K=K,
        stride_am=input.stride(0), stride_ak=input.stride(1),
        stride_bk=weight.stride(1), stride_bn=weight.stride(0),
        stride_bias=bias.stride(0),
        stride_rm=residual.stride(0), stride_rn=residual.stride(1),
        stride_cm=output.stride(0), stride_cn=output.stride(1),
        DTYPE=DTYPE,
    )

    return output


def replacement_func():
    return fused_linear_add_relu