import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul


def replacement_args(in_0, in_1):
    return (in_1, in_0)


# Triton kernel for batched matmul using tl.dot (for N >= 16)
@triton.jit
def matmul_dot_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, M, N, K,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_base = a_ptr + pid_b * stride_a_batch
    b_base = b_ptr + pid_b * stride_b_batch
    c_base = c_ptr + pid_b * stride_c_batch

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
@triton.jit
def matmul_reduce_kernel(
    a_ptr, b_ptr, c_ptr,
    batch_size, M, K, N,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    a_base = a_ptr + pid_b * stride_a_batch
    b_base = b_ptr + pid_b * stride_b_batch
    c_base = c_ptr + pid_b * stride_c_batch

    accumulator = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Load A block: BLOCK_M rows x BLOCK_K cols, cast to float32 for accumulation
        a = tl.load(a_base + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0).to(tl.float32)
        # Load B vector: BLOCK_K elements (N=1), cast to float32
        b = tl.load(b_base + offs_k * stride_b_k,
                     mask=(offs_k < K), other=0.0).to(tl.float32)

        # Element-wise multiply and reduce over K dimension
        accumulator += tl.sum(a * b[None, :], axis=1)

    tl.store(c_base + offs_m * stride_c_m,
             accumulator, mask=(offs_m < M))


@torch.fx.wrap
def batched_matmul_triton(a, b):
    # a: left operand (in_1), b: right operand (in_0)
    # Compute: result = a @ b (batched matmul)

    a = a.contiguous()
    b = b.contiguous()

    # Get dimensions
    orig_shape = list(a.shape[:-2])
    M = a.shape[-2]
    K = a.shape[-1]
    N = b.shape[-1]
    batch_size = 1
    for d in orig_shape:
        batch_size *= d

    # Reshape to 3D for simpler kernel
    a_flat = a.reshape(batch_size, M, K)
    b_flat = b.reshape(batch_size, K, N)

    # Allocate output
    out_flat = torch.empty(batch_size, M, N, dtype=a.dtype, device=a.device)

    if N >= 16:
        # Use tl.dot kernel for larger N
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32

        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N), batch_size)

        matmul_dot_kernel[grid](
            a_ptr=a_flat, b_ptr=b_flat, c_ptr=out_flat,
            batch_size=batch_size, M=M, N=N, K=K,
            stride_a_batch=a_flat.stride(0), stride_a_m=a_flat.stride(1), stride_a_k=a_flat.stride(2),
            stride_b_batch=b_flat.stride(0), stride_b_k=b_flat.stride(1), stride_b_n=b_flat.stride(2),
            stride_c_batch=out_flat.stride(0), stride_c_m=out_flat.stride(1), stride_c_n=out_flat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    else:
        # Use reduce kernel for small N (N=1 etc.)
        BLOCK_M = 64
        BLOCK_K = 64

        grid = (triton.cdiv(M, BLOCK_M), batch_size)

        matmul_reduce_kernel[grid](
            a_ptr=a_flat, b_ptr=b_flat, c_ptr=out_flat,
            batch_size=batch_size, M=M, K=K, N=N,
            stride_a_batch=a_flat.stride(0), stride_a_m=a_flat.stride(1), stride_a_k=a_flat.stride(2),
            stride_b_batch=b_flat.stride(0), stride_b_k=b_flat.stride(1), stride_b_n=b_flat.stride(2),
            stride_c_batch=out_flat.stride(0), stride_c_m=out_flat.stride(1), stride_c_n=out_flat.stride(2),
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
        )

    # Reshape output back to original batch dimensions
    out = out_flat.reshape(*orig_shape, M, N)
    return out


def replacement_func():
    return batched_matmul_triton