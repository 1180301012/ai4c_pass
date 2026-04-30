import torch
import triton
import triton.language as tl


@triton.jit
def batched_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k, stride_b_n,
    stride_c_batch, stride_c_m, stride_c_n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)

    num_pid_n = (N + BLOCK_N - 1) // BLOCK_N
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    a_batch = a_ptr + batch_id * stride_a_batch
    b_batch = b_ptr + batch_id * stride_b_batch
    c_batch = c_ptr + batch_id * stride_c_batch

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_batch + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_batch + offs_k[:, None] * stride_b_k + offs_n[None, :] * stride_b_n
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.dot(a, b)

    c_ptrs = c_batch + offs_m[:, None] * stride_c_m + offs_n[None, :] * stride_c_n
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    if DTYPE == 0:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    elif DTYPE == 1:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


@triton.jit
def batched_gemv_kernel(
    a_ptr, b_ptr, c_ptr,
    M, K,
    stride_a_batch, stride_a_m, stride_a_k,
    stride_b_batch, stride_b_k,
    stride_c_batch, stride_c_m,
    BLOCK_M: tl.constexpr, BLOCK_K: tl.constexpr,
    DTYPE: tl.constexpr,
):
    batch_id = tl.program_id(1)
    pid_m = tl.program_id(0)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)

    a_batch = a_ptr + batch_id * stride_a_batch
    b_batch = b_ptr + batch_id * stride_b_batch
    c_batch = c_ptr + batch_id * stride_c_batch

    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        a_ptrs = a_batch + offs_m[:, None] * stride_a_m + offs_k[None, :] * stride_a_k
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = b_batch + offs_k * stride_b_k
        b_mask = offs_k < K
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc += tl.sum(a.to(tl.float32) * b.to(tl.float32)[None, :], axis=1)

    c_ptrs = c_batch + offs_m * stride_c_m
    c_mask = offs_m < M

    if DTYPE == 0:
        tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)
    elif DTYPE == 1:
        tl.store(c_ptrs, acc.to(tl.bfloat16), mask=c_mask)
    else:
        tl.store(c_ptrs, acc, mask=c_mask)


@torch.fx.wrap
def dispatch_matmul(in_0, in_1, route):
    shape_1 = in_1.shape
    shape_0 = in_0.shape
    ndim = len(shape_1)

    K = shape_1[-1]
    M = shape_1[-2]
    N = shape_0[-1]

    num_batches = 1
    for i in range(ndim - 2):
        num_batches *= shape_1[i]

    stride_a_batch = M * K
    stride_a_m = K
    stride_a_k = 1
    stride_b_batch = K * N
    stride_b_k = N
    stride_b_n = 1
    stride_c_batch = M * N
    stride_c_m = N
    stride_c_n = 1

    out_shape = list(shape_1[:-1]) + [N]
    output = torch.empty(out_shape, device=in_1.device, dtype=in_1.dtype)

    if in_1.dtype == torch.float16:
        dtype_flag = 0
    elif in_1.dtype == torch.bfloat16:
        dtype_flag = 1
    else:
        dtype_flag = 2

    if route == "gemm_op":
        BLOCK_M = 64
        BLOCK_N = 64
        BLOCK_K = 32
        grid = (
            ((M + BLOCK_M - 1) // BLOCK_M) * ((N + BLOCK_N - 1) // BLOCK_N),
            num_batches,
        )
        batched_gemm_kernel[grid](
            in_1, in_0, output,
            M, N, K,
            stride_a_batch, stride_a_m, stride_a_k,
            stride_b_batch, stride_b_k, stride_b_n,
            stride_c_batch, stride_c_m, stride_c_n,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
            DTYPE=dtype_flag,
            num_stages=4, num_warps=4,
        )
    elif route == "gemv_torch":
        BLOCK_M = 32
        # Adaptive BLOCK_K based on K size
        if K <= 64:
            BLOCK_K = 64
            nstages = 1
        elif K <= 256:
            BLOCK_K = 128
            nstages = 2
        else:
            BLOCK_K = 256
            nstages = 2
        grid = (
            (M + BLOCK_M - 1) // BLOCK_M,
            num_batches,
        )
        batched_gemv_kernel[grid](
            in_1, in_0, output,
            M, K,
            stride_a_batch, stride_a_m, stride_a_k,
            stride_b_batch, stride_b_k,
            stride_c_batch, stride_c_m,
            BLOCK_M=BLOCK_M, BLOCK_K=BLOCK_K,
            DTYPE=dtype_flag,
            num_stages=nstages, num_warps=4,
        )

    return output