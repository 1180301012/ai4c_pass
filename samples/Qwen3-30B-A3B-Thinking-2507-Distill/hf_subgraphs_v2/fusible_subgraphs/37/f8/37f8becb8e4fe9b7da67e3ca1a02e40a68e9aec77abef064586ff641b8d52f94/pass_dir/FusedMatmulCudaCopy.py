import torch
import triton
import triton.language as tl
from torch import device


# ---------------------------------------------------------------------------
# Pattern: ONLY the matmul.
# ---------------------------------------------------------------------------

def pattern(in_2, in_3):
    return torch.matmul(in_2, in_3)


def replacement_args(in_2, in_3):
    return (in_2, in_3)


# ---------------------------------------------------------------------------
# Triton GEMV kernel
#
#   Grid = (ceil(M / BLOCK_M),)  — one CTA per BLOCK_M output rows.
#   K is tl.constexpr so Triton specialises the kernel per K value.
#   BLOCK_M=16, BLOCK_K=256.  Float32 accumulation, stored in output dtype.
# ---------------------------------------------------------------------------

@triton.jit
def matmul_gemv_kernel(
    A_ptr, B_ptr, out_ptr,
    M,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    stride_am, stride_ak, stride_bk,
):
    pid    = tl.program_id(0)
    m_offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_offs < M

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        a = tl.load(
            A_ptr + m_offs[:, None] * stride_am + k_offs[None, :] * stride_ak,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr + k_offs * stride_bk,
            mask=k_mask,
            other=0.0,
        )
        acc += tl.sum(a * b[None, :], axis=1)

    tl.store(
        out_ptr + m_offs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask,
    )


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------

_BLOCK_M = 16
_BLOCK_K = 256


@torch.fx.wrap
def triton_matmul_2xK_x_Kx1(in_2, in_3):
    M = in_2.shape[0]
    K = in_2.shape[1]
    out = torch.empty((M, 1), dtype=in_2.dtype, device=in_2.device)

    grid_m = (M + _BLOCK_M - 1) // _BLOCK_M   # = 1 for M=2

    matmul_gemv_kernel[(grid_m,)](
        in_2, in_3, out,
        M, K,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        stride_am=in_2.stride(0),
        stride_ak=in_2.stride(1),
        stride_bk=in_3.stride(0),
        num_warps=2,
        num_stages=3,
    )

    return out


def replacement_func():
    return triton_matmul_2xK_x_Kx1