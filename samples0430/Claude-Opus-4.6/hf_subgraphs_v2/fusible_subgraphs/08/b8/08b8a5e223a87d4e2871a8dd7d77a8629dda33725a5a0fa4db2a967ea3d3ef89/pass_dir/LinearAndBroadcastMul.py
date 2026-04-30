import torch
import triton
import triton.language as tl


def pattern(weight, scale, input_tensor, x):
    linear_out = torch.nn.functional.linear(x, weight, None)
    mul_result = input_tensor * scale
    return mul_result, linear_out


def replacement_args(weight, scale, input_tensor, x):
    return (x, weight, input_tensor, scale)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_matmul_broadcast_mul_kernel(
    x_ptr, w_ptr, inp_ptr, scale_ptr, out_linear_ptr, out_mul_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_im, stride_in,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
    offs_k = tl.arange(0, BLOCK_K)

    # --- Matmul: x @ weight.T ---
    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    b_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        mask_b = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_xk
        b_ptrs += BLOCK_K * stride_wk

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Store linear output (auto-casts from fp32 to output dtype)
    out_l_ptrs = out_linear_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_l_ptrs, acc, mask=mask_out)

    # --- Broadcast multiply: input_tensor * scale ---
    # input_tensor is [M, N], scale is [N]
    inp_ptrs = inp_ptr + offs_m[:, None] * stride_im + offs_n[None, :] * stride_in
    scale_vals = tl.load(scale_ptr + offs_n, mask=(offs_n < N), other=0.0)
    inp_vals = tl.load(inp_ptrs, mask=mask_out, other=0.0)
    mul_result = inp_vals * scale_vals[None, :]

    out_m_ptrs = out_mul_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_m_ptrs, mul_result, mask=mask_out)


@torch.fx.wrap
def linear_and_broadcast_mul_impl(x, weight, input_tensor, scale):
    # x: [B, S, K], weight: [N, K], input_tensor: [B, S, N], scale: [N]
    # Returns: (input_tensor * scale, x @ weight.T)
    K = weight.shape[1]
    N = weight.shape[0]
    M = x.numel() // K

    x_2d = x.reshape(M, K)
    inp_2d = input_tensor.reshape(M, N)

    out_linear = torch.empty((M, N), dtype=x.dtype, device=x.device)
    out_mul = torch.empty((M, N), dtype=input_tensor.dtype, device=input_tensor.device)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    fused_matmul_broadcast_mul_kernel[grid](
        x_2d, weight, inp_2d, scale, out_linear, out_mul,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        inp_2d.stride(0), inp_2d.stride(1),
    )

    # Reshape outputs to match expected shapes
    linear_shape = x.shape[:-1] + (N,)
    mul_shape = input_tensor.shape

    return out_mul.reshape(mul_shape), out_linear.reshape(linear_shape)


def replacement_func():
    return linear_and_broadcast_mul_impl