import torch
import triton
import triton.language as tl

# Generic fused GELU+Add pass that matches ALL shape variants.
# Pattern: gelu(in_2) → flatten(2) → transpose(1,2) → contiguous() → add(in_3)
# Hybrid: 2D-tiled (coalesced) for C=32/large-N, 1D for C=128/256/small-N.

# ── 2D tiled kernel (coalesced for in_2 reads) ───────────────────────────────
# Best for large N (many programs) to saturate GPU and hide memory latency.
@triton.jit
def _gelu_add_2d_kernel(
    in2_ptr, in3_ptr, out_ptr,
    N_VAL: tl.constexpr,
    C_VAL: tl.constexpr,
    TILE: tl.constexpr,      # Square tile size (power of 2)
    N_BLOCKS: tl.constexpr,  # = N_VAL // TILE
):
    pid = tl.program_id(0)
    pid_n = pid % N_BLOCKS
    pid_c = pid // N_BLOCKS
    n_start = pid_n * TILE
    c_start = pid_c * TILE
    n_range = n_start + tl.arange(0, TILE)
    c_range = c_start + tl.arange(0, TILE)

    # Load in_2[c, n] as [TILE, TILE] - consecutive n values → COALESCED
    in2_offs = c_range[:, None] * N_VAL + n_range[None, :]
    in2_raw = tl.load(in2_ptr + in2_offs)
    in2 = in2_raw.to(tl.float32)
    gelu_out = in2 * 0.5 * (1.0 + tl.math.erf(in2 * 0.7071067811865476))

    # Transpose [TILE, TILE] → [TILE, TILE] (square, efficient shared-mem swap)
    gelu_T = tl.trans(gelu_out)

    # Load in_3[n, c] as [TILE, TILE] - consecutive c values → COALESCED
    in3_offs = n_range[:, None] * C_VAL + c_range[None, :]
    in3 = tl.load(in3_ptr + in3_offs).to(tl.float32)

    result = gelu_T + in3

    # Store [TILE, TILE] - consecutive c values → COALESCED
    tl.store(out_ptr + in3_offs, result.to(in2_raw.dtype))


# ── 1D kernel (one program per spatial position) ─────────────────────────────
# Lower overhead per-block; better for small N (C=128/256) where
# the tl.trans shared-memory cost outweighs the coalescing benefit.
@triton.jit
def _gelu_add_1d_kernel(
    in2_ptr, in3_ptr, out_ptr,
    N_VAL: tl.constexpr,
    C_VAL: tl.constexpr,
):
    n = tl.program_id(0)
    c_range = tl.arange(0, C_VAL)

    in2_raw = tl.load(in2_ptr + c_range * N_VAL + n)
    in2 = in2_raw.to(tl.float32)
    gelu_out = in2 * 0.5 * (1.0 + tl.math.erf(in2 * 0.7071067811865476))

    in3 = tl.load(in3_ptr + n * C_VAL + c_range).to(tl.float32)
    result = gelu_out + in3

    tl.store(out_ptr + n * C_VAL + c_range, result.to(in2_raw.dtype))


# ── Pattern ──────────────────────────────────────────────────────────────────
# Matches the shape-agnostic prefix of the model computation.
# Returns tmp_6 (the add result) as the single output.
# The subsequent permute→view→view→permute→layernorm→view remain in graph.
def pattern(in_2, in_3):
    tmp_2 = torch.nn.functional.gelu(in_2, approximate='none')
    tmp_3 = tmp_2.flatten(2)
    tmp_4 = tmp_3.transpose(1, 2)
    tmp_5 = tmp_4.contiguous()
    tmp_6 = in_3 + tmp_5
    return tmp_6


def replacement_args(in_2, in_3):
    return (in_2, in_3)


# ── Replacement ───────────────────────────────────────────────────────────────
@torch.fx.wrap
def _gelu_add_generic(in_2, in_3):
    """Hybrid GELU+Add: 2D-tiled for large N (C=32), 1D for small N (C=128/256)."""
    B, C, H, W = in_2.shape
    N = H * W
    out = torch.empty(B, N, C, dtype=in_2.dtype, device=in_2.device)

    if C == 32:
        # N=3072: 2D tiled → coalesced reads, 96 programs, 4 warps for better occupancy
        N_BLOCKS = N // 32   # = 96
        _gelu_add_2d_kernel[(N_BLOCKS,)](
            in_2, in_3, out,
            N_VAL=N, C_VAL=32, TILE=32, N_BLOCKS=N_BLOCKS, num_warps=4)
    elif C == 128:
        # N=192: 1D simpler, 192 programs, 4 warps
        _gelu_add_1d_kernel[(N,)](
            in_2, in_3, out, N_VAL=N, C_VAL=128, num_warps=4)
    elif C == 256:
        # N=48: 2D tiled with TILE=16 → 3×16=48 programs, better coalescing
        N_BLOCKS = N // 16   # = 3
        C_BLOCKS = 256 // 16  # = 16
        _gelu_add_2d_kernel[(N_BLOCKS * C_BLOCKS,)](
            in_2, in_3, out,
            N_VAL=N, C_VAL=256, TILE=16, N_BLOCKS=N_BLOCKS, num_warps=2)

    return out


def replacement_func():
    return _gelu_add_generic