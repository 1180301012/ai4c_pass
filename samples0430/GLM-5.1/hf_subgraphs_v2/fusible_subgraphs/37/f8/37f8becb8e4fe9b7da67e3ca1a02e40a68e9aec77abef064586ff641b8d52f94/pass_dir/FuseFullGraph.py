import torch
import triton
import triton.language as tl
from torch import device


def pattern(in_0, in_1, in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    tmp_3 = in_1.to(device(type='cuda'))
    tmp_4 = in_0.to(device(type='cuda'))
    return (tmp_4, tmp_3, matmul)


def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    BLOCK_K: tl.constexpr,
):
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
def full_graph_fused(in_0, in_1, in_2, in_3):
    # Compute matmul using Triton kernel
    M, K_a = in_2.shape
    K_b, N = in_3.shape
    K = K_a

    BLOCK_K = 256

    out_matmul = torch.empty((M, N), dtype=in_2.dtype, device=in_2.device)

    grid = (M * N,)

    matmul_kernel[grid](
        a_ptr=in_2, b_ptr=in_3, out_ptr=out_matmul,
        M=M, N=N, K=K,
        stride_am=in_2.stride(0), stride_ak=in_2.stride(1),
        stride_bk=in_3.stride(0), stride_bn=in_3.stride(1),
        stride_om=out_matmul.stride(0), stride_on=out_matmul.stride(1),
        BLOCK_K=BLOCK_K,
        num_warps=2,
    )

    # Transfer scalars to GPU using torch.full (allowed API)
    val_0 = in_0.item()
    val_1 = in_1.item()

    out_0 = torch.full((1,), val_0, dtype=in_0.dtype, device='cuda')
    out_1 = torch.full((1,), val_1, dtype=in_1.dtype, device='cuda')

    return (out_0, out_1, out_matmul)


def replacement_func():
    return full_graph_fused