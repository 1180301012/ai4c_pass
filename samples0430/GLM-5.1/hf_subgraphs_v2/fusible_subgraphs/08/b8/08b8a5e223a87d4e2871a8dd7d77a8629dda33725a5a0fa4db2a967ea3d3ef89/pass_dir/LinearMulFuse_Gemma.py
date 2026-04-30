import torch
import triton
import triton.language as tl


# ==================== Triton Kernels ====================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 2}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_mul_fused_kernel(
    input_ptr, weight_ptr, activation_ptr, output_ptr,
    M, N, K,
    stride_im, stride_ik,
    stride_wk, stride_wn,
    stride_am, stride_an,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        a_ptrs = input_ptr + offs_m[:, None] * stride_im + k[None, :] * stride_ik
        b_ptrs = weight_ptr + k[:, None] * stride_wk + offs_n[None, :] * stride_wn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)

    # Multiply by activation
    act_ptrs = activation_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    act = tl.load(act_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    result = accumulator * act

    # Store result
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(output_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 1}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 2}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 2}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 4}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 4}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        k = k_start + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k[None, :] * stride_ak
        b_ptrs = b_ptr + k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (k[None, :] < K), other=0.0)
        b = tl.load(b_ptrs, mask=(k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        accumulator += tl.dot(a, b, allow_tf32=True)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def broadcast_mul_kernel(
    a_ptr, b_ptr, output_ptr,
    M, N,
    stride_am, stride_an,
    stride_b,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Load a tile: [BLOCK_M, BLOCK_N]
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    a_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    # Load b (1D vector): [BLOCK_N]
    b_ptrs = b_ptr + offs_n * stride_b
    b_mask = offs_n < N
    b = tl.load(b_ptrs, mask=b_mask, other=0.0)

    # Multiply with broadcasting: output[m, n] = a[m, n] * b[n]
    result = a * b

    # Store
    output_ptrs = output_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    output_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(output_ptrs, result, mask=output_mask)


# ==================== Wrapper Functions ====================

@torch.fx.wrap
def _linear_mul_fuse_impl(weight, input, activation):
    M = input.shape[0] * input.shape[1]
    K = input.shape[-1]
    N = weight.shape[0]

    output = torch.empty((input.shape[0], input.shape[1], N), dtype=input.dtype, device=input.device)

    stride_im = input.stride(-2)
    stride_ik = input.stride(-1)
    stride_wk = weight.stride(1)
    stride_wn = weight.stride(0)
    stride_am = activation.stride(-2)
    stride_an = activation.stride(-1)
    stride_om = output.stride(-2)
    stride_on = output.stride(-1)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    linear_mul_fused_kernel[grid](
        input_ptr=input, weight_ptr=weight, activation_ptr=activation, output_ptr=output,
        M=M, N=N, K=K,
        stride_im=stride_im, stride_ik=stride_ik,
        stride_wk=stride_wk, stride_wn=stride_wn,
        stride_am=stride_am, stride_an=stride_an,
        stride_om=stride_om, stride_on=stride_on,
    )

    return output


@torch.fx.wrap
def _matmul_impl(x, weight):
    M = x.shape[0] * x.shape[1]
    K = x.shape[-1]
    N = weight.shape[0]

    output = torch.empty((x.shape[0], x.shape[1], N), dtype=x.dtype, device=x.device)

    stride_am = x.stride(-2)
    stride_ak = x.stride(-1)
    stride_bk = weight.stride(1)
    stride_bn = weight.stride(0)
    stride_cm = output.stride(-2)
    stride_cn = output.stride(-1)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a_ptr=x, b_ptr=weight, c_ptr=output,
        M=M, N=N, K=K,
        stride_am=stride_am, stride_ak=stride_ak,
        stride_bk=stride_bk, stride_bn=stride_bn,
        stride_cm=stride_cm, stride_cn=stride_cn,
    )

    return output


@torch.fx.wrap
def _broadcast_mul_impl(input, scale):
    M = input.shape[0] * input.shape[1]
    N = input.shape[-1]

    output = torch.empty_like(input)

    stride_am = input.stride(-2)
    stride_an = input.stride(-1)
    stride_b = scale.stride(0)
    stride_om = output.stride(-2)
    stride_on = output.stride(-1)

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']),
        triton.cdiv(N, META['BLOCK_N']),
    )

    broadcast_mul_kernel[grid](
        a_ptr=input, b_ptr=scale, output_ptr=output,
        M=M, N=N,
        stride_am=stride_am, stride_an=stride_an,
        stride_b=stride_b,
        stride_om=stride_om, stride_on=stride_on,
    )

    return output


# ==================== Dispatch Wrapper ====================

@torch.fx.wrap
def dispatch_wrapper(*args):
    route = args[-1]
    if route == "linear_mul_fuse":
        weight, input_tensor, activation = args[0], args[1], args[2]
        result = _linear_mul_fuse_impl(weight, input_tensor, activation)
        return (result,)
    elif route == "independent_ops":
        weight, scale, input_tensor, x = args[0], args[1], args[2], args[3]
        mul_result = _broadcast_mul_impl(input_tensor, scale)
        linear_result = _matmul_impl(x, weight)
        return (mul_result, linear_result)
    else:
        raise ValueError(f"Unknown route: {route}")


# ==================== Pattern (Gemma-specific) ====================

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_0, None)
    tmp_2 = in_1 * linear
    return (tmp_2,)


# ==================== Replacement Args ====================

def replacement_args(in_0, in_1, in_2):
    # in_0 = weight, in_1 = activation, in_2 = input
    # Reorder to: weight, input, activation
    return (in_0, in_2, in_1, "linear_mul_fuse")


# ==================== Replacement Func ====================

def replacement_func():
    return dispatch_wrapper