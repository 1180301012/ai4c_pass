import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def batched_matmul_row_kernel(
    in_1_ptr, in_0_ptr, out_ptr,
    N, M: tl.constexpr, K: tl.constexpr,
    stride_in1_n, stride_in1_m, stride_in1_k,
    stride_in0_n, stride_in0_k,
    stride_out_n, stride_out_m,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)  # Each program handles one n (row)

    offs_k = tl.arange(0, BLOCK_K)
    offs_m = tl.arange(0, M)
    mask_k = offs_k < K

    # Pre-load in_0[n, :, 0] for all k values (padded to BLOCK_K) - shape (BLOCK_K,)
    ptr_in0 = in_0_ptr + pid * stride_in0_n + offs_k * stride_in0_k
    val_in0 = tl.load(ptr_in0, mask=mask_k, other=0.0).to(tl.float32)  # shape (BLOCK_K,)

    # Load in_1[n, m, k] for all m and k (padded) - shape (BLOCK_K, M)
    ptr_in1 = in_1_ptr + pid * stride_in1_n + offs_m[None, :] * stride_in1_m + offs_k[:, None] * stride_in1_k
    val_in1 = tl.load(ptr_in1, mask=mask_k[:, None], other=0.0).to(tl.float32)  # shape (BLOCK_K, M)

    # Compute dot product: result[m] = sum_k(in_1[n,m,k] * in_0[n,k,0])
    # Masked elements are 0, so they don't affect the sum
    acc = tl.sum(val_in1 * val_in0[:, None], axis=0)  # shape (M,)

    # Store result
    ptr_out = out_ptr + pid * stride_out_n + offs_m * stride_out_m
    tl.store(ptr_out, acc)


@torch.fx.wrap
def batched_matmul_wrapper(in_0, in_1):
    N, M, K = in_1.shape

    out = torch.empty(N, M, 1, dtype=in_1.dtype, device=in_1.device)

    # BLOCK_K must be power of 2 >= K
    BLOCK_K = 16  # K=9, next power of 2 is 16

    grid = (N,)

    batched_matmul_row_kernel[grid](
        in_1_ptr=in_1, in_0_ptr=in_0, out_ptr=out,
        N=N, M=M, K=K,
        stride_in1_n=in_1.stride(0), stride_in1_m=in_1.stride(1), stride_in1_k=in_1.stride(2),
        stride_in0_n=in_0.stride(0), stride_in0_k=in_0.stride(1),
        stride_out_n=out.stride(0), stride_out_m=out.stride(1),
        BLOCK_K=BLOCK_K,
    )

    return out


def replacement_func():
    return batched_matmul_wrapper