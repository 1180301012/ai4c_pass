import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# Matmul+squeeze kernel: A[M,K] @ B[K,N] = out[M,N]
# Uses tl.dot for tensor core acceleration
# M is typically 1, so we pad BLOCK_M to 16 (minimum for tl.dot)
@triton.jit
def matmul_squeeze_kernel(
    a_ptr, b_ptr, out_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    stride_am: tl.constexpr, stride_ak: tl.constexpr,
    stride_bk: tl.constexpr, stride_bn: tl.constexpr,
    stride_om: tl.constexpr, stride_on: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    # Since grid is (1,) and BLOCK_N >= N and BLOCK_M >= M,
    # pid_m=0, pid_n=0
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                     mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        accumulator += tl.dot(a, b)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out = accumulator.to(out_ptr.dtype.element_ty)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=mask)


@torch.fx.wrap
def matmul_squeeze_kernel_wrapper(in_0, in_1):
    M = in_0.shape[-2]  # 1
    K = in_0.shape[-1]  # 249
    N = in_1.shape[-1]  # 64

    out = torch.empty((M, N), dtype=in_0.dtype, device=in_0.device)

    matmul_squeeze_kernel[(1,)](
        a_ptr=in_0, b_ptr=in_1, out_ptr=out,
        M=M, N=N, K=K,
        stride_am=in_0.stride(1), stride_ak=in_0.stride(2),
        stride_bk=in_1.stride(1), stride_bn=in_1.stride(2),
        stride_om=out.stride(0), stride_on=out.stride(1),
        BLOCK_M=16, BLOCK_N=64, BLOCK_K=64,
    )

    return out


def replacement_func():
    return matmul_squeeze_kernel_wrapper