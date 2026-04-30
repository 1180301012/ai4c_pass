import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_1, in_0)
    return matmul


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def batched_matvec_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    BM,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < BM

    b = offsets // M
    m = offsets % M

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    base_0 = b * K
    base_1 = b * (M * K) + m * K

    for k in range(K):
        v0 = tl.load(in_0_ptr + base_0 + k, mask=mask, other=0.0)
        v1 = tl.load(in_1_ptr + base_1 + k, mask=mask, other=0.0)
        acc += v0.to(tl.float32) * v1.to(tl.float32)

    tl.store(out_ptr + offsets, acc.to(out_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def batched_matvec(in_0, in_1):
    B = in_1.shape[0]
    M = in_1.shape[1]
    K = in_1.shape[2]
    BM = B * M

    out = torch.empty(B, M, 1, device=in_0.device, dtype=in_0.dtype)

    BLOCK_SIZE = 128
    grid = ((BM + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    batched_matvec_kernel[grid](
        in_0, in_1, out,
        BM,
        M=M, K=K,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=1,
        num_stages=1,
    )

    return out


def replacement_func():
    return batched_matvec