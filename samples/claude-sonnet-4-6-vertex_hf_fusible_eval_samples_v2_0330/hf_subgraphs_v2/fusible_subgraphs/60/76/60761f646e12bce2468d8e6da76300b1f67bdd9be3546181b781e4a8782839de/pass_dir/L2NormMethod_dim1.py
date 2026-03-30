import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: row-wise L2 normalization (available for future use)
# ---------------------------------------------------------------------------
@triton.jit
def l2_normalize_kernel(
    x_ptr, out_ptr, N, stride_row,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_offset = row_idx * stride_row
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)
    x_f32 = x.to(tl.float32)
    sq_sum = tl.sum(x_f32 * x_f32, 0)
    norm = tl.sqrt(sq_sum)
    norm = tl.maximum(norm, 1e-12)
    out = (x_f32 / norm).to(x.dtype)
    tl.store(out_ptr + row_offset + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Replacement: eliminate the no-op cat; the downstream normalize still runs.
# Replacing cat([x], 1) with x.contiguous() removes the redundant copy and
# lets torch.compile fuse the subsequent normalize more efficiently.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_cat_normalize(x: torch.Tensor) -> torch.Tensor:
    """Replace cat([x], 1) with a contiguous view — identical semantics, no copy."""
    return x.contiguous()


# ---------------------------------------------------------------------------
# Pattern / replacement interface
# ---------------------------------------------------------------------------
def pattern(x):
    return torch.cat([x], 1)


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_cat_normalize