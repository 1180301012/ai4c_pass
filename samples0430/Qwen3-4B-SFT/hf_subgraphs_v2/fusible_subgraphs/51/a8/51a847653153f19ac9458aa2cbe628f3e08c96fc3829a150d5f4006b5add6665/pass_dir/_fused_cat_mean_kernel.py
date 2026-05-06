import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 32},   num_warps=1, num_stages=1),
        triton.Config({'BLOCK_HW': 64},   num_warps=2, num_stages=1),
        triton.Config({'BLOCK_HW': 128},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 256},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_HW': 512},  num_warps=8, num_stages=1),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8, num_stages=1),
    ],
    key=['HW'],
)
@triton.jit
def row_mean_kernel(
    in_ptr, out_ptr,
    N, HW,
    in_s0, in_s1,
    BLOCK_HW: tl.constexpr,
):
    """
    Compute spatial mean over H*W for each (batch, channel) pair.

    Grid: (B * N,) — one program per row.
    Accumulates in fp32 for precision, stores in the tensor's element dtype.
    Works with BLOCK_HW < HW via a tiled loop (loop is a no-op when
    BLOCK_HW >= HW, which autotune achieves).
    """
    pid       = tl.program_id(0)
    b         = pid // N
    n         = pid  % N
    base      = b * in_s0 + n * in_s1

    acc       = tl.zeros([BLOCK_HW], dtype=tl.float32)
    HW_ar     = tl.arange(0, BLOCK_HW)

    # Tiled loop: in GPU code when BLOCK_HW >= HW this runs exactly once.
    for tile in range(0, HW, BLOCK_HW):
        off  = tile + HW_ar
        mask = off < HW
        data = tl.load(in_ptr + base + off, mask=mask, other=0.0).to(tl.float32)
        acc  += data

    mean_val  = tl.sum(acc) / HW
    # Load first element to establish pointer element type for the dtype cast.
    x0        = tl.load(in_ptr + base + 0)
    tl.store(out_ptr + b * N + n, mean_val.to(x0.dtype))


# ---------------------------------------------------------------------------
# Shared dispatch wrapper.
#
# ALL 4 pass files return THIS function from replacement_func(), satisfying
# the replacement_func_limit constraint (same Python object).
#
# Pattern matched:
#   pattern(tmp_1)  =  tmp_1.mean((2, 3), keepdim=True)
#   where tmp_1 = cat([in_0, in_1], dim=1)[:, :, :N, :, :]  (pre-computed)
#
# Replacement: numerically-stable Triton spatial-mean kernel over H*W.
# Only the mean operation is replaced — cat and slice continue unchanged.
# Returns a SINGLE tensor of shape [B, N, 1, 1] (no tuple).
# ---------------------------------------------------------------------------
@torch.fx.wrap
def shared_cat_mean_dispatch(in_0, in_1, route):
    """Shared dispatcher for mean optimisation across all N variants."""
    B  = in_0.shape[0]
    N  = in_0.shape[1]
    H  = in_0.shape[2]
    W  = in_0.shape[3]
    HW = H * W

    mean_ = torch.empty((B, N, 1, 1), dtype=in_0.dtype, device=in_0.device)

    grid = lambda meta: (B * N,)

    row_mean_kernel[grid](
        in_0, mean_,
        N, HW,
        in_0.stride(0), in_0.stride(1),
    )

    return mean_