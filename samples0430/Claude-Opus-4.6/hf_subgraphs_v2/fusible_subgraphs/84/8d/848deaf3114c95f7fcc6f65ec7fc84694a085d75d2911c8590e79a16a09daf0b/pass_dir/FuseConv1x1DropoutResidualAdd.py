import torch
import triton
import triton.language as tl


def pattern(bias, weight, residual, input_tensor):
    conv_out = torch.conv2d(input_tensor, weight, bias, (1, 1), (0, 0), (1, 1), 1)
    drop_out = torch.nn.functional.dropout(conv_out, 0.0, False, False)
    result = drop_out + residual
    return result


def replacement_args(bias, weight, residual, input_tensor):
    return (bias, weight, residual, input_tensor)


@triton.autotune(
    configs=[
        # BLOCK_K=64: 4 K iterations
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        # BLOCK_K=128: 2 K iterations
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_conv1x1_bias_residual_kernel(
    weight_ptr, input_ptr, bias_ptr, residual_ptr, output_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    # Program ID with L2-friendly grouping
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

    # Pointer setup using constexpr strides: weight[M,K] stride=[K,1], input[K,N] stride=[N,1]
    a_ptrs = weight_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = input_ptr + offs_k[:, None] * N + offs_n[None, :]

    # GEMM loop - fully unrolled by compiler (K/BLOCK_K is constexpr)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    # Fused epilogue: bias + residual add
    bias_val = tl.load(bias_ptr + offs_m)
    acc += bias_val[:, None]

    r_ptrs = residual_ptr + offs_m[:, None] * N + offs_n[None, :]
    acc += tl.load(r_ptrs)

    o_ptrs = output_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(o_ptrs, acc)


@torch.fx.wrap
def fused_conv1x1_bias_residual(bias, weight, residual, input_tensor):
    M = weight.shape[0]
    K = weight.shape[1]
    N = input_tensor.shape[2] * input_tensor.shape[3]

    output = torch.empty_like(residual)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    fused_conv1x1_bias_residual_kernel[grid](
        weight, input_tensor, bias, residual, output,
        M, N, K,
    )

    return output


def replacement_func():
    return fused_conv1x1_bias_residual