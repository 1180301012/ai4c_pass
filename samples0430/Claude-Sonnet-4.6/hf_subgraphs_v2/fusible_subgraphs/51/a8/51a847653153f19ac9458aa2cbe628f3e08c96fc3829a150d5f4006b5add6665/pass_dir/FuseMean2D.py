"""
FuseMean2D: Replace x.mean((2, 3), keepdim=True) with a Triton kernel.

Single-output pattern: only matches the spatial mean, so
    len(match.returning_nodes) == len(copied_returning_nodes) == 1.
No multi-output assert issues.
"""
import torch
import triton
import triton.language as tl


# ── Triton kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64},   num_warps=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['C', 'HW'],
)
@triton.jit
def _mean_hw_kernel(
    src_ptr, out_ptr,
    C, HW,
    src_bs, src_cs,   # batch / channel stride of src  (elements)
    BLOCK_SIZE: tl.constexpr,
):
    """
    Grid: (B * C,).  pid = b * C + c.
    Reduces src[b, c, :, :] (HW elements) → out[b, c, 0, 0].
    out is laid out as [B, C, 1, 1] contiguous → flat idx = b*C + c.
    Accumulation is in float32 for precision; Triton auto-casts on store.
    """
    pid = tl.program_id(0)
    b   = pid // C
    c   = pid % C

    src_base = src_ptr + b * src_bs + c * src_cs

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x    = tl.load(src_base + offs, mask=mask, other=0.0)
        acc  = acc + x.to(tl.float32)

    total    = tl.sum(acc, axis=0)
    mean_val = total / HW
    tl.store(out_ptr + b * C + c, mean_val)


# ── @torch.fx.wrap wrapper (opaque to FX) ────────────────────────────────────

@torch.fx.wrap
def triton_mean_2d(x):
    """Spatial mean over dims (2, 3) with keepdim=True, Triton-accelerated."""
    B  = x.shape[0]
    C  = x.shape[1]
    H  = x.shape[2]
    W  = x.shape[3]
    HW = H * W

    out = torch.empty([B, C, 1, 1], dtype=x.dtype, device=x.device)

    _mean_hw_kernel[(B * C,)](
        x, out,
        C, HW,
        x.stride(0), x.stride(1),
    )
    return out


# ── Pass interface ─────────────────────────────────────────────────────────────

def pattern(x):
    y = x.mean((2, 3), keepdim=True)
    return y


def replacement_args(x):
    return (x,)


def replacement_func():
    return triton_mean_2d