import torch
import triton
import triton.language as tl


def pattern(in_0):
    """
    Match: transpose(1,2) → contiguous → reshape → contiguous
    in_0: matmul output [B, H, M, D] (any dtype)
    cuBLAS does the matmul; we just fuse the layout transformation.
    """
    tmp_6 = in_0.transpose(1, 2)
    tmp_7 = tmp_6.contiguous()
    tmp_8 = tmp_7.reshape(1, 257, -1)
    tmp_9 = tmp_8.contiguous()
    return tmp_9


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 128}, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_D': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_D': 64},  num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_D': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_D': 64},  num_warps=4),
    ],
    key=['M', 'D'],
)
@triton.jit
def transpose_reshape_kernel(
    In_ptr, Out_ptr,
    stride_ib, stride_ih, stride_im, stride_id,
    B, H, M, D,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy [B,H,M,D] → [B,M,H*D] (fused transpose + reshape)."""
    bh = tl.program_id(0)
    m_blk = tl.program_id(1)

    b = bh // H
    h = bh % H

    m_start = m_blk * BLOCK_M
    m_range = m_start + tl.arange(0, BLOCK_M)
    d_range = tl.arange(0, BLOCK_D)

    # Load In [BLOCK_M, BLOCK_D] from [B, H, M, D] layout
    In_ptrs = (In_ptr + b * stride_ib + h * stride_ih
               + m_range[:, None] * stride_im + d_range[None, :] * stride_id)
    in_mask = (m_range[:, None] < M) & (d_range[None, :] < D)
    data = tl.load(In_ptrs, mask=in_mask, other=0.0)

    # Write to [B, M, H*D] layout
    out_base = Out_ptr + b * (M * H * D)
    out_ptrs = out_base + m_range[:, None] * (H * D) + h * D + d_range[None, :]
    out_mask = (m_range[:, None] < M) & (d_range[None, :] < D)
    tl.store(out_ptrs, data, mask=out_mask)


@torch.fx.wrap
def fused_av_matmul_wrapper(X):
    """
    X: matmul output [B, H, M, D] (any dtype)
    Returns: [B, M, H*D] — fused transpose + reshape
    """
    B, H, M, D = X.shape
    Out = torch.empty((B, M, H * D), dtype=X.dtype, device=X.device)

    grid = lambda meta: (B * H, triton.cdiv(M, meta['BLOCK_M']))

    transpose_reshape_kernel[grid](
        X, Out,
        X.stride(0), X.stride(1), X.stride(2), X.stride(3),
        B, H, M, D,
    )
    return Out


def replacement_func():
    return fused_av_matmul_wrapper