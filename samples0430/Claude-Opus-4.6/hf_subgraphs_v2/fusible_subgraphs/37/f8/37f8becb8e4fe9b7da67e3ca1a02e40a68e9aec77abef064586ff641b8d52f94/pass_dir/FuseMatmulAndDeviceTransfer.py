import torch
import triton
import triton.language as tl


def pattern(in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    return matmul


def replacement_args(in_2, in_3):
    return (in_2, in_3)


@triton.jit
def matvec_kernel(
    in_2_ptr, in_3_ptr, out_ptr,
    K,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    k_offsets = tl.arange(0, BLOCK_K)
    mask = k_offsets < K

    a = tl.load(in_2_ptr + row * K + k_offsets, mask=mask, other=0.0)
    b = tl.load(in_3_ptr + k_offsets, mask=mask, other=0.0)

    result = tl.sum(a.to(tl.float32) * b.to(tl.float32), axis=0)

    tl.store(out_ptr + row, result)


@torch.fx.wrap
def optimized_matmul(in_2, in_3):
    M = in_2.shape[0]
    K = in_2.shape[1]
    out = torch.empty((M, 1), dtype=in_2.dtype, device=in_2.device)
    BLOCK_K = triton.next_power_of_2(K)
    matvec_kernel[(M,)](in_2, in_3, out, K, BLOCK_K=BLOCK_K)
    return out


def replacement_func():
    return optimized_matmul