import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: match torch.nn.functional.normalize(x, p=2, dim=1)
# The upstream torch.cat([in_0], 1) is a no-op (single-tensor concat),
# so the L2-normalisation is applied to the same data regardless of whether
# the cat node is still present in the graph or was folded away.
# We replace normalize with a fast Triton L2-normalise kernel.
# ---------------------------------------------------------------------------

def pattern(x):
    return torch.nn.functional.normalize(x, p=2, dim=1)


def replacement_args(x):
    return (x,)


# ---------------------------------------------------------------------------
# Triton kernel – one program per row
#   1. Load all N floats of a row into registers
#   2. Accumulate sum-of-squares in fp32 for numerical stability
#   3. rsqrt(norm²) → per-element multiply
#   4. Store back in original dtype
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def _l2_normalize_kernel(
    x_ptr,
    out_ptr,
    N,                       # number of columns (row width)
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    row_start = row * N

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load one row (masked loads return 0.0 for out-of-bound lanes)
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)

    # Promote to fp32 to avoid precision loss in sum-of-squares accumulation
    x_f32 = x.to(tl.float32)
    norm_sq = tl.sum(x_f32 * x_f32, axis=0)
    inv_norm = tl.rsqrt(norm_sq)

    # Normalise and cast back to original dtype before storing
    out = (x_f32 * inv_norm).to(x.dtype)
    tl.store(out_ptr + row_start + offsets, out, mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper – invoked in place of the matched subgraph
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_l2_normalize(x):
    """
    Drop-in replacement for torch.nn.functional.normalize(x, p=2, dim=1).
    One Triton program handles one row (one sample / feature-vector pair).
    """
    B = x.shape[0]
    N = x.shape[1]

    out = torch.empty_like(x)

    # Grid = (B,); BLOCK_SIZE is resolved by the autotuner
    grid = lambda meta: (B,)
    _l2_normalize_kernel[grid](x, out, N)

    return out


def replacement_func():
    return triton_l2_normalize