import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# torch.sym_sum is a symbolic-shapes helper introduced in newer PyTorch.
# Patch it in at import time so eager baseline execution doesn't crash.
# The result is immediately discarded in every model (tmp_3 = None), so the
# exact semantics don't matter for correctness.
# ---------------------------------------------------------------------------
if not hasattr(torch, 'sym_sum'):
    def _sym_sum(args):
        result = args[0]
        for a in args[1:]:
            result = result + a
        return result
    torch.sym_sum = _sym_sum


# ---------------------------------------------------------------------------
# Pattern: SiLU (inplace) -> spatial mean (dims 2,3) -> view(1,1,-1)
# The in_0 integer path is dead code and excluded from the pattern.
# ---------------------------------------------------------------------------

def pattern(in_1):
    tmp_0 = torch.nn.functional.silu(in_1, inplace=True)
    tmp_1 = tmp_0.mean((2, 3))
    tmp_4 = tmp_1.view(1, 1, -1)
    return (tmp_0, tmp_4)


def replacement_args(in_1):
    return (in_1,)


# ---------------------------------------------------------------------------
# Triton kernel: fused SiLU + spatial-mean reduction
#
# Grid: (N*C,)  — one program per (batch, channel) slice.
# Each program iterates over H*W elements in BLOCK_SIZE-wide tiles,
# applies SiLU in-place, and accumulates a float32 sum for the mean.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}),
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
        triton.Config({'BLOCK_SIZE': 512}),
        triton.Config({'BLOCK_SIZE': 1024}),
        triton.Config({'BLOCK_SIZE': 2048}),
    ],
    key=['HW'],
)
@triton.jit
def _silu_mean_kernel(
    x_ptr,      # [N, C, H, W]  — modified in-place to hold SiLU(x)
    mean_ptr,   # [N, C]         — output: spatial mean of SiLU(x)
    HW,         # H * W  (runtime)
    BLOCK_SIZE: tl.constexpr,
):
    nc_idx = tl.program_id(0)
    base   = nc_idx * HW

    # Accumulator in fp32 for numerical stability
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW

        # Load (fp16/bf16/fp32)
        x_raw  = tl.load(x_ptr + base + offsets, mask=mask, other=0.0)
        x_f32  = x_raw.to(tl.float32)

        # SiLU: x * sigmoid(x)
        silu_x = x_f32 * tl.sigmoid(x_f32)

        # Write SiLU result back (Triton auto-casts fp32 → original dtype)
        tl.store(x_ptr + base + offsets, silu_x, mask=mask)

        # Accumulate (masked positions contribute 0)
        acc += tl.where(mask, silu_x, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))

    # Reduce and store mean (Triton auto-casts fp32 → output dtype)
    total    = tl.sum(acc)
    mean_val = total / HW
    tl.store(mean_ptr + nc_idx, mean_val)


# ---------------------------------------------------------------------------
# Python wrapper (must be @torch.fx.wrap so FX doesn't try to trace into it)
# ---------------------------------------------------------------------------

@torch.fx.wrap
def silu_mean_fused(x):
    """
    Fused SiLU + spatial-mean kernel.

    Args:
        x : Tensor [N, C, H, W]  (float16 / bfloat16 / float32)

    Returns:
        (x, tmp4) where
          x    is modified in-place with SiLU(x)   shape [N, C, H, W]
          tmp4 is the spatial mean viewed as         shape [1, 1, N*C]
    """
    N, C, H, W = x.shape
    NC = N * C
    HW = H * W

    # Output for mean — same dtype as input to match PyTorch semantics
    mean_out = torch.empty((N, C), dtype=x.dtype, device=x.device)

    _silu_mean_kernel[(NC,)](
        x, mean_out,
        HW,
        # BLOCK_SIZE is selected by autotune
    )

    tmp4 = mean_out.view(1, 1, -1)
    return (x, tmp4)


# ---------------------------------------------------------------------------
# replacement_func: zero-argument function returning the callable
# ---------------------------------------------------------------------------

def replacement_func():
    return silu_mean_fused