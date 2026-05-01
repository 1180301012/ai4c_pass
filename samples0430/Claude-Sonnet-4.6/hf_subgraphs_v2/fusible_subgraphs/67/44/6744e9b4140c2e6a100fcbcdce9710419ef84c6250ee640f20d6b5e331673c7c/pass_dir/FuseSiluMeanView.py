import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Pattern: Python-level "call_method: mean" with dim=(2,3)
#
# We match ONLY the mean reduction (not silu) because:
#  - The compiled graph is at the torch-function level (not aten).
#  - ForceArgsTracer does NOT modify call_method nodes, so (2,3) is kept.
#  - Silu has kwargs={'inplace':True} in the compiled graph but ForceArgsTracer
#    moves them to positional args in the pattern, causing arg-count mismatch.
#
# By matching only mean, the silu and view nodes stay untouched in the graph:
#   silu(in_1)               → tmp_0   [unchanged, left in graph]
#   fast_spatial_mean(tmp_0) → tmp_1   [our Triton kernel]
#   tmp_1.view(1, 1, -1)     → tmp_4   [unchanged, left in graph]
#   return (tmp_0, tmp_4)
# ---------------------------------------------------------------------------
def pattern(in_1):
    return in_1.mean((2, 3))


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: single-pass parallel reduction using tl.sum (warp shuffles).
# BLOCK_SIZE = next_power_of_2(HW) so the entire channel is reduced in ONE
# shot without a sequential loop — far lower latency than a loop-based kernel.
#
# One program per (b, c) pair; the grid is (B*C,).
# ---------------------------------------------------------------------------
@triton.jit
def _spatial_mean_kernel(
    x_ptr,
    out_ptr,
    HW,
    BLOCK_SIZE: tl.constexpr,   # must be >= HW; caller uses next_power_of_2(HW)
):
    pid     = tl.program_id(0)
    base    = pid * HW
    offsets = tl.arange(0, BLOCK_SIZE)
    mask    = offsets < HW
    x       = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
    total   = tl.sum(x, axis=0)           # single warp-shuffle reduction
    tl.store(out_ptr + pid, total / HW)   # auto-converts fp32 → out dtype


@torch.fx.wrap
def fast_spatial_mean(in_1):
    """Triton-backed mean over spatial dims (H, W). Returns [B, C] tensor."""
    B, C, H, W = in_1.shape
    HW         = H * W
    BLOCK_SIZE = triton.next_power_of_2(HW)   # >= HW, power-of-2 for warp reductions

    out = torch.empty((B, C), dtype=in_1.dtype, device=in_1.device)
    _spatial_mean_kernel[(B * C,)](in_1, out, HW=HW, BLOCK_SIZE=BLOCK_SIZE)
    return out


def replacement_func():
    return fast_spatial_mean