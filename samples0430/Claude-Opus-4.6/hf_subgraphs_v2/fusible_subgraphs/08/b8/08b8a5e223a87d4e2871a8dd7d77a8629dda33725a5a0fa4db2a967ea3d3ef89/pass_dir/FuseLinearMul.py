import torch
import triton
import triton.language as tl


def pattern(weight, gate, x):
    linear_out = torch.nn.functional.linear(x, weight, None)
    result = gate * linear_out
    return result


def replacement_args(weight, gate, x):
    return (x, weight, gate)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 1}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_SIZE_M': 4}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_mul_kernel(
    x_ptr, w_ptr, gate_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_gm, stride_gn,
    stride_om, stride_on,
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

    # Pointers for x [M, K] and weight.T [K, N]
    # weight is [N, K], we want weight.T[k, n] = weight[n, k]
    a_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    b_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Accumulate matmul in float32
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

    # Fused gate multiply: result = gate * linear_out
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    gate_ptrs = gate_ptr + offs_m[:, None] * stride_gm + offs_n[None, :] * stride_gn
    gate = tl.load(gate_ptrs, mask=mask_out, other=0.0)

    # Cast accumulator to gate dtype then multiply (matches PyTorch behavior)
    result = gate * acc.to(gate.dtype)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=mask_out)


@torch.fx.wrap
def fused_linear_mul_impl(x, weight, gate):
    # x: [..., K], weight: [N, K], gate: [..., N]
    # output: [..., N] = gate * (x @ weight.T)
    orig_shape = gate.shape
    K = weight.shape[1]
    N = weight.shape[0]
    M = x.numel() // K

    x_2d = x.reshape(M, K)
    gate_2d = gate.reshape(M, N)
    out = torch.empty_like(gate_2d)

    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)

    fused_linear_mul_kernel[grid](
        x_2d, weight, gate_2d, out,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(0), weight.stride(1),
        gate_2d.stride(0), gate_2d.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(orig_shape)


def replacement_func():
    return fused_linear_mul_impl