import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.cat([in_0], 1)
    tmp_1 = torch.nn.functional.normalize(tmp_0, p=2, dim=1)
    return tmp_1


def replacement_args(in_0):
    return (in_0,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 1024}, num_warps=4),
        triton.Config({'BLOCK_D': 1024}, num_warps=8),
        triton.Config({'BLOCK_D': 1024}, num_warps=16),
        triton.Config({'BLOCK_D': 2048}, num_warps=8),
        triton.Config({'BLOCK_D': 2048}, num_warps=16),
    ],
    key=['N', 'D'],
)
@triton.jit
def _l2_normalize_kernel(
    x_ptr, out_ptr,
    N, D,
    x_stride_n, x_stride_d,
    eps,
    BLOCK_D: tl.constexpr,
):
    """One program per row: load row, compute L2 norm, store normalised row."""
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    # Load row from (possibly non-contiguous) input
    x = tl.load(
        x_ptr + row_idx * x_stride_n + offsets * x_stride_d,
        mask=mask, other=0.0,
    )

    # Accumulate in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    sum_sq = tl.sum(x_f32 * x_f32, axis=0)
    norm = tl.sqrt(sum_sq)
    norm = tl.maximum(norm, eps)

    # Normalise and cast back to original dtype
    out = (x_f32 / norm).to(x.dtype)

    # Store to contiguous output (row-major, stride = D)
    tl.store(out_ptr + row_idx * D + offsets, out, mask=mask)


@torch.fx.wrap
def fused_l2_normalize(in_0):
    N, D = in_0.shape
    # Always produce a contiguous output, matching cat([x], 1) + normalize semantics
    out = torch.empty((N, D), dtype=in_0.dtype, device=in_0.device)
    eps = 1e-12

    _l2_normalize_kernel[(N,)](
        in_0, out,
        N, D,
        in_0.stride(0), in_0.stride(1),
        eps,
    )
    return out


def replacement_func():
    return fused_l2_normalize