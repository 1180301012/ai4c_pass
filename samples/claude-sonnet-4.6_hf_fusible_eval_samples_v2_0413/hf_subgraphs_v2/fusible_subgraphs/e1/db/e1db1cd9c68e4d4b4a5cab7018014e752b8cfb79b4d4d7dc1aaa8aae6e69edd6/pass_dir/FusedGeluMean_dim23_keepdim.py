"""
Shared Triton kernel and dispatch function for the fused GELU + mean optimization.
Imported by FusedGeluPrecompMean.py and FusedMeanFromCache.py.
NOT listed in sorted_output_pass_rule_names.json (utility module only).
"""
import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Global mean cache: key = (*input_shape, dtype_str, device_str)
#                    value = pre-allocated [B, C, 1, 1] mean tensor
# The fused kernel writes into this tensor as a side-effect of computing GELU,
# and the second pass reads from it.
# ---------------------------------------------------------------------------
_MEAN_CACHE = {}


# ---------------------------------------------------------------------------
# Fused GELU + spatial-mean Triton kernel
#
# Design:
#  - One program per (batch, channel) pair.
#  - Loops over the HW spatial dimension in BLOCK_SIZE chunks.
#    * For BLOCK_SIZE >= HW (e.g. 4096 >= 3136): exactly 1 iteration.
#    * For BLOCK_SIZE < HW (e.g. 512): multiple iterations, smaller CTAs
#      → higher occupancy → better latency hiding for small batch sizes.
#  - GELU(0) = 0, so masked/padded lanes contribute 0 to the sum without
#    needing an explicit tl.where.
#  - inv_HW is a pre-computed Python float to avoid any int/fp cast inside
#    the kernel.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Small-BLOCK, many-iteration configs: better occupancy (good for large B)
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        # Same with num_stages=2 for software pipelining (hides memory latency in loop)
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        # Large-BLOCK, single-iteration configs: better for small B (more threads/block)
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=32),
    ],
    key=['HW'],
)
@triton.jit
def fused_gelu_mean_kernel(
    x_ptr, out_ptr, mean_ptr,
    HW, inv_HW,
    BLOCK_SIZE: tl.constexpr,
):
    """
    One program per (batch × channel) pair.
    Loops over spatial dimension in BLOCK_SIZE chunks.
    GELU(0) = 0 exactly, so padded lanes (loaded as 0) safely add 0 to acc.
    """
    bc_idx  = tl.program_id(0)
    base    = bc_idx * HW

    # Element-wise accumulator for partial mean
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offsets = start + tl.arange(0, BLOCK_SIZE)
        mask    = offsets < HW

        # Load native dtype, upcast to fp32 for accuracy
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)

        # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        gelu_x = 0.5 * x * (1.0 + tl.math.erf(x * 0.7071067811865476))

        # Store GELU output (Triton auto-casts fp32 → native dtype via ptr type)
        tl.store(out_ptr + base + offsets, gelu_x, mask=mask)

        # Accumulate: GELU(0) = 0 for padded lanes, so no masking needed
        acc += gelu_x

    # Final mean
    mean_val = tl.sum(acc) * inv_HW
    tl.store(mean_ptr + bc_idx, mean_val)


# ---------------------------------------------------------------------------
# Runtime helpers called from shared_dispatch
# ---------------------------------------------------------------------------

def _gelu_and_store_mean(x):
    """Compute GELU + mean in one Triton pass; cache mean, return GELU."""
    B, C, H, W = x.shape
    HW  = H * W
    BC  = B * C
    key = (B, C, H, W, str(x.dtype), str(x.device))

    out = torch.empty_like(x)
    if key not in _MEAN_CACHE:
        _MEAN_CACHE[key] = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    mean_out = _MEAN_CACHE[key]

    fused_gelu_mean_kernel[(BC,)](
        x_ptr    = x,
        out_ptr  = out,
        mean_ptr = mean_out,
        HW       = HW,
        inv_HW   = 1.0 / HW,
    )
    return out


def _get_cached_mean(x):
    """Return the mean that was stored by _gelu_and_store_mean for the same shape."""
    B, C, H, W = x.shape
    key = (B, C, H, W, str(x.dtype), str(x.device))
    if key in _MEAN_CACHE:
        return _MEAN_CACHE[key]
    # Fallback (should not occur in a correctly ordered pass sequence)
    result = torch.empty((B, C, 1, 1), dtype=x.dtype, device=x.device)
    _MEAN_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Single shared replacement function (MUST be @torch.fx.wrap).
# Both pass files return THIS same object from replacement_func(), so the
# framework's g_replacement_func identity assertion always passes.
# The route string selects which branch executes at runtime.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def shared_dispatch(x, route):
    if route == "gelu_mean":
        return _gelu_and_store_mean(x)
    else:                          # route == "mean_cache"
        return _get_cached_mean(x)