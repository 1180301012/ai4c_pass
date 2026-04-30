import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    return matmul


def replacement_args(in_2, in_3):
    return (in_2, in_3)


@triton.jit
def matmul_m2_n1_kernel(
    a_ptr, b_ptr, out_ptr,
    K,
    stride_am, stride_ak,
    stride_bk,
    stride_om,
    BLOCK_K: tl.constexpr,
):
    # Specialized for M=2, N=1: single program computes both output elements
    # Shares b loading across both rows
    acc0 = 0.0
    acc1 = 0.0

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load b[k_offs] - shared between both rows (col=0)
        b = tl.load(b_ptr + k_offs * stride_bk, mask=k_mask, other=0.0).to(tl.float32)

        # Load a[0, k_offs] and a[1, k_offs]
        a0 = tl.load(a_ptr + k_offs * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        a1 = tl.load(a_ptr + stride_am + k_offs * stride_ak, mask=k_mask, other=0.0).to(tl.float32)

        acc0 += tl.sum(a0 * b)
        acc1 += tl.sum(a1 * b)

    # Store results (row=0,col=0 and row=1,col=0)
    tl.store(out_ptr, acc0)
    tl.store(out_ptr + stride_om, acc1)


@triton.jit
def matmul_general_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_K: tl.constexpr,
):
    # General kernel: each program computes one output element
    pid = tl.program_id(0)
    row = pid // N
    col = pid % N

    if row >= M:
        return

    acc = 0.0

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a = tl.load(a_ptr + row * stride_am + k_offs * stride_ak, mask=k_mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + k_offs * stride_bk + col * stride_bn, mask=k_mask, other=0.0).to(tl.float32)

        acc += tl.sum(a * b)

    tl.store(out_ptr + row * stride_om + col * stride_on, acc)


@torch.fx.wrap
def triton_matmul(a, b):
    M, K_a = a.shape
    K_b, N = b.shape
    assert K_a == K_b
    K = K_a

    BLOCK_K = 256

    out = torch.empty((M, N), dtype=a.dtype, device=a.device)

    # Use specialized kernel for M=2, N=1 (our evaluation cases)
    if M == 2 and N == 1:
        matmul_m2_n1_kernel[(1,)](
            a_ptr=a, b_ptr=b, out_ptr=out,
            K=K,
            stride_am=a.stride(0), stride_ak=a.stride(1),
            stride_bk=b.stride(0),
            stride_om=out.stride(0),
            BLOCK_K=512,
            num_warps=1,
        )
    else:
        # Fallback to general kernel
        matmul_general_kernel[(M * N,)](
            a_ptr=a, b_ptr=b, out_ptr=out,
            M=M, N=N, K=K,
            stride_am=a.stride(0), stride_ak=a.stride(1),
            stride_bk=b.stride(0), stride_bn=b.stride(1),
            stride_om=out.stride(0), stride_on=out.stride(1),
            BLOCK_K=BLOCK_K,
            num_warps=2,
        )

    return out


def replacement_func():
    return triton_matmul