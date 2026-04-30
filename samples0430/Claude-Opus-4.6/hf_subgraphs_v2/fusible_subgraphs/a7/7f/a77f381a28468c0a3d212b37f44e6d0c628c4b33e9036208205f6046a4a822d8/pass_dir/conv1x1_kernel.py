import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_s1_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """1x1 conv with stride 1 as batched GEMM.
    M = C_out, N = H*W, K = C_in
    weight: [M, K] row-major
    input: [B, K, N] where each [K, N] slice is row-major
    output: [B, M, N] row-major
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Weight: weight[m, k] = weight_ptr + m * K + k
    a_ptrs = weight_ptr + offs_m[:, None] * K + offs_k[None, :]
    # Input: input[b, k, n] = input_ptr + b * K * N + k * N + n
    b_ptrs = input_ptr + pid_b * (K * N) + offs_k[:, None] * N + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * N

    # Output: output[b, m, n] = output_ptr + b * M * N + m * N + n
    c_ptrs = output_ptr + pid_b * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_s2_kernel(
    input_ptr, weight_ptr, output_ptr,
    M, N, K,
    HW_in, W_in, W_out,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """1x1 conv with stride 2 as batched GEMM with gather.
    M = C_out, N = H_out*W_out, K = C_in
    """
    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Map output spatial index to input spatial index (stride 2)
    h_out = offs_n // W_out
    w_out = offs_n % W_out
    inp_spatial = (2 * h_out) * W_in + (2 * w_out)

    # Weight: weight[m, k] = weight_ptr + m * K + k
    a_ptrs = weight_ptr + offs_m[:, None] * K + offs_k[None, :]
    # Input: input[b, k, spatial] = input_ptr + b * K * HW_in + k * HW_in + spatial
    b_ptrs = input_ptr + pid_b * (K * HW_in) + offs_k[:, None] * HW_in + inp_spatial[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_remaining = K - k_start
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K
        b_ptrs += BLOCK_K * HW_in

    # Output: output[b, m, n] = output_ptr + b * M * N + m * N + n
    c_ptrs = output_ptr + pid_b * (M * N) + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(output_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def dispatch_conv1x1(in_0, in_1, stride_val, slice_n, order_flag):
    """Dispatch function for 1x1 conv + slice.
    
    Args:
        in_0: weight [C_out, C_in, 1, 1]
        in_1: input [B, C_in, H, W]
        stride_val: 1 or 2
        slice_n: number of channels to slice
        order_flag: 0 = (slice, full), 1 = (full, slice)
    """
    weight = in_0
    input_tensor = in_1

    B = input_tensor.shape[0]
    C_in = input_tensor.shape[1]
    H = input_tensor.shape[2]
    W = input_tensor.shape[3]
    C_out = weight.shape[0]

    if stride_val == 1:
        H_out = H
        W_out = W
    else:
        H_out = H // stride_val
        W_out = W // stride_val

    N = H_out * W_out
    M = C_out
    K = C_in

    output = torch.empty((B, C_out, H_out, W_out), device=input_tensor.device, dtype=input_tensor.dtype)

    grid = lambda meta: (
        triton.cdiv(N, meta['BLOCK_N']),
        triton.cdiv(M, meta['BLOCK_M']),
        B,
    )

    if stride_val == 1:
        _conv1x1_s1_kernel[grid](
            input_tensor, weight, output,
            M, N, K,
        )
    else:
        HW_in = H * W
        _conv1x1_s2_kernel[grid](
            input_tensor, weight, output,
            M, N, K,
            HW_in, W, W_out,
        )

    slice_output = output[:, :slice_n, :, :]

    if order_flag == 0:
        return (slice_output, output)
    else:
        return (output, slice_output)