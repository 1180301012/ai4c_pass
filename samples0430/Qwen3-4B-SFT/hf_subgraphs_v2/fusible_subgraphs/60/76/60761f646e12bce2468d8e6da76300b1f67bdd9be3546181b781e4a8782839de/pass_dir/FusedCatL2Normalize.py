import torch
import triton
import triton.language as tl


def pattern(x):
    cat_result = torch.cat([x], 1)
    normalized = torch.nn.functional.normalize(cat_result, p=2, dim=1)
    return normalized


def replacement_args(x):
    return (x,)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['D'],
)
@triton.jit
def _l2_normalize_kernel(
    x_ptr,
    out_ptr,
    D,
    stride_row,
    others: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One program per row. Each program normalizes one row of length D to L2 unit."""
    row_idx = tl.program_id(0)
    base = x_ptr + row_idx * stride_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < D
    x = tl.load(base + offsets, mask=mask, other=others)

    # Compute L2 norm in float32 for numerical stability
    x_f32 = x.to(tl.float32)
    norm_sq = tl.sum(x_f32 * x_f32, axis=0)
    inv_norm = tl.rsqrt(norm_sq)

    # Scale and convert back to original dtype
    out = (x_f32 * inv_norm).to(x.dtype)
    tl.store(out_ptr + row_idx * stride_row + offsets, out, mask=mask)


@torch.fx.wrap
def triton_cat_normalize(x):
    """Fused cat(…, dim=1) + L2 normalize.  Uses single kernel to skip the
    intermediate allocation from cat, saving 1 full memory round-trip."""
    # cat([x], dim=1) on a 2-D tensor [B, D] is effectively a no-op; the
    # output shape / contiguity is the same as x.
    out = torch.empty_like(x)
    B = x.shape[0]
    D = x.shape[1]

    _l2_normalize_kernel[(B,)](
        x,
        out,
        D,
        x.stride(0),
        0.0,
    )
    return out


def replacement_func():
    return triton_cat_normalize