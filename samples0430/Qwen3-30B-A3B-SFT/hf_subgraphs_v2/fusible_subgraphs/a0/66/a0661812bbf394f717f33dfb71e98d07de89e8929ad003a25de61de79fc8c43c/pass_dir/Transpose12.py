"""
Pass: Transpose12
Matches: any .transpose(1, 2) call → single tensor output
Applied after linear+dropout passes so it sees the fused output.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def _transpose12_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Transpose dims 1 and 2 of a [B, M, N] tensor → [B, N, M].
    Treating the tensor as [M, N] per batch element (B treated as outer),
    we copy x[b, m, n] → out[b, n, m].
    We iterate over batch * M as a flat row index.
    """
    pid = tl.program_id(0)
    total_rows = M * N
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_rows

    # 2-D coordinates in the [M, N] matrix
    m = offs // N
    n = offs % N

    # Load x[b, m, n]  (flat index: batch_idx * M*N + m*N + n)
    # We treat the tensor as [total_rows] flat per batch layer.
    vals = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # Store to out[b, n, m]: index = b*(N*M) + n*M + m
    # We need to map the flat offs = m*N+n to n*M+m.
    # First recover b = offs // (M*N)
    out_offs = (offs // total_rows) * (N * M) + n * M + m
    tl.store(out_ptr + out_offs, vals, mask=mask)


@torch.fx.wrap
def triton_transpose12(x):
    """
    Compute x.transpose(1, 2) using a Triton kernel.
    x: [B, M, N]  →  out: [B, N, M]
    """
    B = x.shape[0]
    M = x.shape[1]
    N = x.shape[2]

    out = torch.empty((B, N, M), dtype=x.dtype, device=x.device)

    # For batch dimension: launch one grid per batch element
    # The kernel treats each batch's [M, N] block independently
    grid = lambda META: (B * triton.cdiv(M * N, META['BLOCK_SIZE']),)

    _transpose12_kernel[grid](
        x, out,
        M, N,
    )

    return out