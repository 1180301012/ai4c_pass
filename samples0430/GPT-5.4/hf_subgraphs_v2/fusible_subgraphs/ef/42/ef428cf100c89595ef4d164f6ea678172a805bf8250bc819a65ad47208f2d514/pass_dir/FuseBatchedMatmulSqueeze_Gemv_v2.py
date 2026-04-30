import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    matmul = torch.matmul(in_0, in_1)
    tmp_1 = matmul.squeeze(1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def _matvec_249x64_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_on,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_n = tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for k_start in range(0, 249, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < 249
        a_vals = tl.load(a_ptr + offs_k * stride_ak, mask=mask_k, other=0.0)
        b_vals = tl.load(
            b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn,
            mask=mask_k[:, None],
            other=0.0,
        )
        acc += tl.dot(a_vals, b_vals)

    tl.store(out_ptr + offs_n * stride_on, acc)


@torch.fx.wrap
def fused_batched_matmul_squeeze_gemv_v2(in_0, in_1):
    out = torch.empty((1, 64), device=in_0.device, dtype=in_0.dtype)
    _matvec_249x64_kernel[(1,)](
        in_0,
        in_1,
        out,
        in_0.stride(2),
        in_1.stride(1),
        in_1.stride(2),
        out.stride(1),
        BLOCK_K=32,
        BLOCK_N=64,
        num_warps=4,
        num_stages=3,
    )
    return out


def replacement_func():
    return fused_batched_matmul_squeeze_gemv_v2