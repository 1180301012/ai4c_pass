import torch
import triton
import triton.language as tl

from torch import device


# Pattern matching function
def pattern(in_2, in_3):
    matmul = torch.matmul(in_2, in_3)
    return matmul


# Argument extraction function
def replacement_args(in_2, in_3):
    return (in_2, in_3)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 2, "BLOCK_K": 128}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_M": 2, "BLOCK_K": 256}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 128}, num_warps=1, num_stages=1),
        triton.Config({"BLOCK_M": 4, "BLOCK_K": 256}, num_warps=2, num_stages=1),
        triton.Config({"BLOCK_M": 8, "BLOCK_K": 256}, num_warps=2, num_stages=1),
    ],
    key=["M", "K"],
)
@triton.jit
def _siglip_gemv_n1_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_cm,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k_block in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        a = tl.load(
            a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        b = tl.load(
            b_ptr + offs_k * stride_bk,
            mask=offs_k < K,
            other=0.0,
        )
        acc += tl.sum(a.to(tl.float32) * b[None, :].to(tl.float32), axis=1)

    tl.store(c_ptr + offs_m * stride_cm, acc, mask=offs_m < M)


@torch.fx.wrap
def _siglip_matmul_only(in_2, in_3):
    m = in_2.shape[0]
    n = in_3.shape[1]
    out = torch.empty((m, n), device=in_2.device, dtype=in_2.dtype)

    if out.numel() != 0:
        grid = lambda META: (triton.cdiv(m, META["BLOCK_M"]),)
        _siglip_gemv_n1_kernel[grid](
            in_2,
            in_3,
            out,
            m,
            in_2.shape[1],
            in_2.stride(0),
            in_2.stride(1),
            in_3.stride(0),
            out.stride(0),
        )

    return out


# Replacement function (NO arguments, returns function reference)
def replacement_func():
    return _siglip_matmul_only