import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel: fused cat (no-op, single tensor) + L2-normalize along dim=1
# One Triton program per row.  N=768 always → BLOCK_N=1024 covers it in one
# shot (256 lanes are masked to 0 so they don't pollute the sum-of-squares).
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_warps=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
        triton.Config({'BLOCK_N': 2048}, num_warps=4),
        triton.Config({'BLOCK_N': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def l2_normalize_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_row,
    BLOCK_N: tl.constexpr,
):
    """
    One program per row.
    Each program loads BLOCK_N elements (with masking), computes the L2 norm
    in float32 for numerical stability, clamps to eps=1e-12, then stores the
    normalised values back in the original dtype.
    """
    row_idx = tl.program_id(0)
    row_offset = row_idx * stride_row

    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    # ---- load ---------------------------------------------------------------
    x = tl.load(x_ptr + row_offset + cols, mask=mask, other=0.0)

    # ---- compute L2 norm in fp32 for precision ------------------------------
    x_f32 = x.to(tl.float32)
    sq_sum = tl.sum(x_f32 * x_f32, 0)
    norm = tl.sqrt(sq_sum)
    # Same eps as torch.nn.functional.normalize default
    norm = tl.maximum(norm, 1e-12)

    # ---- normalize and cast back to original dtype --------------------------
    out = (x_f32 / norm).to(x.dtype)

    # ---- store (only valid columns) -----------------------------------------
    tl.store(out_ptr + row_offset + cols, out, mask=mask)


# ---------------------------------------------------------------------------
# Host wrapper (must be decorated with @torch.fx.wrap so the graph rewriter
# treats the call as a single opaque node).
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_cat_l2_normalize(x: torch.Tensor) -> torch.Tensor:
    """
    Semantically equivalent to:
        tmp = torch.cat([x], 1)          # identity for single-tensor cat
        return torch.nn.functional.normalize(tmp, p=2, dim=1)
    """
    x_cont = x.contiguous()
    M, N = x_cont.shape[0], x_cont.shape[1]

    out = torch.empty_like(x_cont)

    # One program per row
    grid = (M,)

    l2_normalize_kernel[grid](
        x_cont,
        out,
        M,
        N,
        x_cont.stride(0),      # stride between rows (= N for contiguous)
    )
    return out


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C pass framework
# ---------------------------------------------------------------------------

def pattern(x):
    """
    Decomposed L2-normalize without expand_as (relies on broadcasting):
        norm = linalg.vector_norm(x, 2, [1], keepdim=True)
        return x / clamp(norm, min=1e-12)
    Matches the torch._decomp decomposition of normalize(p=2, dim=1).
    """
    norm = torch.linalg.vector_norm(x, 2, dim=[1], keepdim=True)
    clamped = norm.clamp_min(1e-12)
    return x / clamped


def replacement_args(x):
    return (x,)


def replacement_func():
    return fused_cat_l2_normalize