# Shared Triton kernels and kernel wrapper for batched matmul optimization
import torch
import triton
import triton.language as tl


# Triton kernel for batched matmul using tl.dot (for N >= 16)
# Handles 4D tensors with separate B and H dimension strides
@triton.jit
def matmul_dot_kernel_4d(
    a_ptr, b_ptr, c_ptr,
    B_dim, H_dim, M, N, K,
    stride_a_b, stride_a_h, stride_a_m, stride_a_k,
    stride_b_b, stride_b_h, stride_b_k, stride_b_n,
    stride_c_b, stride_c_h, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_bh = tl.program_id(2)

    # Decompose batch-head index into separate B and H indices
    b_idx = pid_bh // H_dim
    h_idx = pid_bh % H_dim

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Compute base offsets for this batch element
    a_base = a_ptr + b_idx * stride_a_b + h_idx * stride_a_h
    b_base = b_ptr + b_idx * stride_b_b + h_idx * stride_b_h
    c_base = c_ptr + b_idx * stride_c_b + h_idx * stride_c_h

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a = tl.load(a_base + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_base + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        accumulator += tl.dot(a, b, allow_tf32=False)

    tl.store(c_base + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n,
             accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# Triton kernel for batched matrix-vector product (for small N, especially N=1)
# Handles 4D tensors with separate B and H dimension strides
@triton.jit
def matmul_reduce_kernel_4d(
    a_ptr, b_ptr, c_ptr,
    B_dim, H_dim, M, K, N,
    stride_a_b, stride_a_h, stride_a_m, stride_a_k,
    stride_b_b, stride_b_h, stride_b_k, stride_b_n,
    stride_c_b, stride_c_h, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    # Decompose batch-head index into separate B and H indices
    b_idx = pid_bh // H_dim
    h_idx = pid_bh % H_dim

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    # Compute base offsets for this batch element
    a_base = a_ptr + b_idx * stride_a_b + h_idx * stride_a_h
    b_base = b_ptr + b_idx * stride_b_b + h_idx * stride_b_h
    c_base = c_ptr + b_idx * stride_c_b + h_idx * stride_c_h

    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A block: BLOCK_M rows x BLOCK_K cols, cast to float32 for accumulation
        a = tl.load(a_base + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        # Load B vector: BLOCK_K elements (N is small), cast to float32
        b = tl.load(b_base + offs_k * stride_b_k,
                     mask=(offs_k < K), other=0.0).to(tl.float32)

        # Element-wise multiply and reduce over K dimension
        accumulator += tl.sum(a * b[None, :], axis=1)

    tl.store(c_base + offs_m * stride_c_m,
             accumulator, mask=(offs_m < M))


@torch.fx.wrap
def batched_matmul_triton(a, b):
    # a: left operand (in_1), shape [B, H, M, K]
    # b: right operand (in_0), shape [B, H, K, N]
    # Compute: result = a @ b (batched matmul)
    # No reshape/contiguous - handle 4D strides directly in kernel

    B_dim = a.shape[0]
    H_dim = a.shape[1]
    M = a.shape[2]
    K = a.shape[3]
    N = b.shape[3]

    batch_size = B_dim * H_dim

    # Allocate output using torch.empty (allowed API only)
    out = torch.empty(B_dim, H_dim, M, N, dtype=a.dtype, device=a.device)

    if N >= 16:
        # Use tl.dot kernel for larger N
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch_size)

        matmul_dot_kernel_4d[grid](
            a_ptr=a, b_ptr=b, c_ptr=out,
            B_dim=B_dim, H_dim=H_dim, M=M, N=N, K=K,
            stride_a_b=a.stride(0), stride_a_h=a.stride(1), stride_a_m=a.stride(2), stride_a_k=a.stride(3),
            stride_b_b=b.stride(0), stride_b_h=b.stride(1), stride_b_k=b.stride(2), stride_b_n=b.stride(3),
            stride_c_b=out.stride(0), stride_c_h=out.stride(1), stride_c_m=out.stride(2), stride_c_n=out.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    else:
        # Use reduce kernel for small N (N=1 etc.)
        BLOCK_M = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), batch_size)

        matmul_reduce_kernel_4d[grid](
            a_ptr=a, b_ptr=b, c_ptr=out,
            B_dim=B_dim, H_dim=H_dim, M=M, K=K, N=N,
            stride_a_b=a.stride(0), stride_a_h=a.stride(1), stride_a_m=a.stride(2), stride_a_k=a.stride(3),
            stride_b_b=b.stride(0), stride_b_h=b.stride(1), stride_b_k=b.stride(2), stride_b_n=b.stride(3),
            stride_c_b=out.stride(0), stride_c_h=out.stride(1), stride_c_m=out.stride(2), stride_c_n=out.stride(3),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

    return out