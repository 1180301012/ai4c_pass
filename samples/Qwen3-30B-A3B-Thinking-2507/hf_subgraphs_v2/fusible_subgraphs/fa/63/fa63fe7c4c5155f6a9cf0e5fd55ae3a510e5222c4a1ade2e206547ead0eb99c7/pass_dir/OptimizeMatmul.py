import torch
import triton
import triton.language as tl

@triton.jit
def batched_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    batch_dim, M, K, N,
    block_M: tl.constexpr, block_N: tl.constexpr, block_K: tl.constexpr,
    stride_A_batch, stride_A_M, stride_A_K,
    stride_B_batch, stride_B_K, stride_B_N,
    stride_C_batch, stride_C_M, stride_C_N
):
    # Grid: (batch_dim, M//block_M, N//block_N)
    block_id = tl.program_id(0)
    batch = block_id // (M // block_M * N // block_N)
    m_block = (block_id % (M // block_M * N // block_N)) // (N // block_N)
    n_block = block_id % (N // block_N)

    # Allocate shared memory for A and B tiles
    # No shared memory - using global memory for tiles

    # Compute output tile start
    m_start = m_block * block_M
    n_start = n_block * block_N

    # Initialize accumulator
    acc = tl.zeros((block_M, block_N), dtype=tl.float32)

    # Loop over K dimension
    for k in range(0, K, block_K):
        # Load A tile into shared memory
        a_ptrs = A_ptr + batch * stride_A_batch + m_start * stride_A_M + k * stride_A_K
        a = tl.load(
            a_ptrs + tl.arange(0, block_M)[:, None] * stride_A_M + tl.arange(0, block_K)[None, :] * stride_A_K,
            mask=(m_start + tl.arange(0, block_M) < M)[:, None] & (k + tl.arange(0, block_K) < K)[None, :],
            other=0.0
        )


        # Load B tile into shared memory
        b_ptrs = B_ptr + batch * stride_B_batch + k * stride_B_K + n_start * stride_B_N
        b = tl.load(
            b_ptrs + tl.arange(0, block_K)[:, None] * stride_B_K + tl.arange(0, block_N)[None, :] * stride_B_N,
            mask=(k + tl.arange(0, block_K) < K)[:, None] & (n_start + tl.arange(0, block_N) < N)[None, :],
            other=0.0
        )


        # Synchronize to ensure A and B tiles are loaded


        # Compute dot product using shared memory
        acc += tl.dot(a, b)

    # Store result
    c_ptrs = C_ptr + batch * stride_C_batch + m_start * stride_C_M + n_start * stride_C_N
    tl.store(
        c_ptrs + tl.arange(0, block_M)[:, None] * stride_C_M + tl.arange(0, block_N)[None, :] * stride_C_N,
        acc,
        mask=(m_start + tl.arange(0, block_M) < M)[:, None] & (n_start + tl.arange(0, block_N) < N)[None, :]
    )

@torch.fx.wrap
def optimized_matmul(x, y):
    # Get dimensions
    B, H, M, K = x.shape
    _, _, K, N = y.shape

    # Total batch for kernel = B * H
    batch_dim = B * H
    assert N == K, "Output head dimension mismatch"

    # Configure kernel blocks
    block_M, block_N, block_K = 16, 16, 16
    grid_M = (M + block_M - 1) // block_M
    grid_N = (N + block_N - 1) // block_N
    grid = (batch_dim * grid_M * grid_N,)

    # Allocate output
    out = torch.empty((B, H, M, N), dtype=x.dtype, device=x.device)

    # Launch kernel
    batched_matmul_kernel[grid](
        x, y, out,
        batch_dim, M, K, N,
        block_M, block_N, block_K,
        x.stride(0), x.stride(2), x.stride(3),
        y.stride(0), y.stride(2), y.stride(3),
        out.stride(0), out.stride(2), out.stride(3)
    )
    return out

def pattern(in_1, in_0):
    return in_1 @ in_0

def replacement_args(in_1, in_0):
    return (in_1, in_0)

def replacement_func():
    return optimized_matmul