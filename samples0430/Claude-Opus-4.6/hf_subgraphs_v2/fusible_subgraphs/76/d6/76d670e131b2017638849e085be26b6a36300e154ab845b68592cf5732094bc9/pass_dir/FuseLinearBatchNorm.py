import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    linear = torch.nn.functional.linear(in_6, in_5, in_4)
    tmp_7 = torch.nn.functional.batch_norm(in_7, in_0, in_1, in_3, in_2, False, 0.1, 1e-05)
    return (linear, tmp_7)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5, in_6, in_7):
    return (in_6, in_5, in_4, in_7, in_0, in_1, in_3, in_2)


# Batch norm eval kernel - very lightweight
@triton.jit
def bn_eval_kernel(
    input_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, output_ptr,
    N, C,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_C)
    mask = cols < C

    mean = tl.load(mean_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)

    scale = w / tl.sqrt(var + 1e-5)
    offset = b - mean * scale

    offs = row * C + cols
    x = tl.load(input_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x * scale + offset
    tl.store(output_ptr + offs, y, mask=mask)


# GEMM kernel for linear
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_am[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_bn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    bias = tl.load(bias_ptr + offs_bn, mask=offs_bn < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@torch.fx.wrap
def fused_linear_bn(input_lin, weight, bias_lin, input_bn, running_mean, running_var, bn_weight, bn_bias):
    # Linear: input_lin @ weight.T + bias_lin
    M, K = input_lin.shape
    N = weight.shape[0]
    output_lin = torch.empty((M, N), device=input_lin.device, dtype=input_lin.dtype)

    grid_lin = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid_lin](
        input_lin, weight, bias_lin, output_lin,
        M, N, K,
        input_lin.stride(0), input_lin.stride(1),
        weight.stride(1), weight.stride(0),
        output_lin.stride(0), output_lin.stride(1),
    )

    # Batch norm eval
    N_bn = input_bn.shape[0]
    C = input_bn.shape[1]
    output_bn = torch.empty_like(input_bn)
    BLOCK_C = triton.next_power_of_2(C)

    bn_eval_kernel[(N_bn,)](
        input_bn, running_mean, running_var, bn_weight, bn_bias, output_bn,
        N_bn, C,
        BLOCK_C=BLOCK_C,
    )

    return (output_lin, output_bn)


def replacement_func():
    return fused_linear_bn