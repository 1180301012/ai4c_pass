import torch
import triton
import triton.language as tl


# ─── Pattern ────────────────────────────────────────────────────────────────
# Match the spatial mean reduction.
# Both z (relu output) and mean_out (spatial mean) are returned by the model,
# so both must appear as pattern outputs.

def pattern(z):
    return z.mean((2, 3), keepdim=True)


def replacement_args(z):
    return (z,)


# ─── Triton kernel ───────────────────────────────────────────────────────────
# Grid: (N*C,) — one program per (batch, channel) slice.
# Computes: mean of z over the H*W spatial dimensions.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=2),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=16),
    ],
    key=['HW'],
)
@triton.jit
def _spatial_mean_kernel(
    z_ptr,
    out_mean_ptr,
    HW,
    BLOCK_HW: tl.constexpr,
):
    pid  = tl.program_id(0)
    base = pid * HW
    acc  = tl.zeros([BLOCK_HW], dtype=tl.float32)

    for offset in range(0, HW, BLOCK_HW):
        offs = offset + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        zv   = tl.load(z_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc  = acc + tl.where(mask, zv, 0.0)

    mean_val = tl.sum(acc) / HW
    tl.store(out_mean_ptr + pid, mean_val)


# ─── Kernel wrapper ──────────────────────────────────────────────────────────

@torch.fx.wrap
def triton_spatial_mean(z):
    """Compute spatial mean over (H,W) — replaces z.mean((2,3),keepdim=True)."""
    N, C, H, W = z.shape
    HW = H * W

    out_mean_flat = torch.empty((N * C,), dtype=z.dtype, device=z.device)

    _spatial_mean_kernel[(N * C,)](
        z,
        out_mean_flat,
        HW,
    )

    return out_mean_flat.view(N, C, 1, 1)


# ─── Replacement entry point ─────────────────────────────────────────────────

def replacement_func():
    return triton_spatial_mean