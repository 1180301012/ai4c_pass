import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Kernel 1: LayerNorm + GELU, one CTA per row, writes contiguous [B,M,N]
#
# - 3999 CTAs -> high occupancy (fills all SMs)
# - Coalesced reads (input stride-1 in N) and coalesced writes (temp stride-1 in N)
# - No strided access at all
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def _ln_gelu_1d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    """
    input  / output : [B, M, N]  (same contiguous layout)
    One CTA per row; computes LayerNorm then GELU in float32.
    """
    row_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < N

    # Load row -> compute in float32 for precision
    x = tl.load(input_ptr + row_idx * N + offsets,
                 mask=mask, other=0.0).to(tl.float32)

    # LayerNorm: mean + biased variance
    mean = tl.sum(x, axis=0) / N
    diff = x - mean
    var  = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + 1e-5)
    x_norm = diff * rstd

    weight = tl.load(weight_ptr + offsets, mask=mask, other=1.0).to(tl.float32)
    bias   = tl.load(bias_ptr   + offsets, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * weight + bias

    # Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    INV_SQRT2 = 0.7071067811865476
    y_gelu = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    # Coalesced write to temp (same row layout as input)
    tl.store(output_ptr + row_idx * N + offsets, y_gelu, mask=mask)


# ---------------------------------------------------------------------------
# Kernel 2: Tiled coalesced transpose  [B,M,N] -> [B,N,M]
#
# Reads  [TILE_M, TILE_N] tiles from src  (stride-1 in N -> coalesced)
# Writes [TILE_N, TILE_M] tiles to dst   (stride-1 in M via tl.trans -> coalesced)
# Only ~8-16 float16 per thread -> minimal register pressure, high occupancy.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'TILE_M': 16, 'TILE_N': 16}, num_warps=2),
        triton.Config({'TILE_M': 16, 'TILE_N': 32}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 32}, num_warps=4),
        triton.Config({'TILE_M': 32, 'TILE_N': 64}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 32}, num_warps=4),
        triton.Config({'TILE_M': 64, 'TILE_N': 64}, num_warps=8),
        triton.Config({'TILE_M': 128, 'TILE_N': 32}, num_warps=8),
        triton.Config({'TILE_M': 32, 'TILE_N': 128}, num_warps=8),
    ],
    key=['M', 'N'],
)
@triton.jit
def _tiled_transpose_kernel(
    src_ptr,
    dst_ptr,
    B,
    M,
    N,
    TILE_M: tl.constexpr,
    TILE_N: tl.constexpr,
):
    """
    src : [B, M, N]  contiguous
    dst : [B, N, M]  contiguous
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    m_offs = pid_m * TILE_M + tl.arange(0, TILE_M)
    n_offs = pid_n * TILE_N + tl.arange(0, TILE_N)

    m_mask = m_offs < M
    n_mask = n_offs < N

    # Coalesced read [TILE_M, TILE_N] (stride-1 in N)
    src_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    tile = tl.load(src_ptr + src_offs,
                   mask=m_mask[:, None] & n_mask[None, :],
                   other=0.0)

    # Coalesced write [TILE_N, TILE_M] (stride-1 in M via tl.trans)
    dst_offs = pid_b * N * M + n_offs[:, None] * M + m_offs[None, :]
    tl.store(dst_ptr + dst_offs,
             tl.trans(tile),
             mask=n_mask[:, None] & m_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [N]         float16/bfloat16
    in_1 : weight [N]         float16/bfloat16
    in_2 : input  [B, M, N]   float16/bfloat16
    Returns [B, N, M]: layer_norm -> transpose(-2,-1) -> gelu

    Two-step Triton implementation:
      Step 1: _ln_gelu_1d_kernel  (3999 CTAs, high occupancy, coalesced I/O)
              LayerNorm + GELU -> contiguous temp [B, M, N]
      Step 2: _tiled_transpose_kernel  (2000 CTAs, coalesced via tl.trans)
              temp [B, M, N] -> output [B, N, M]
    """
    B, M, N = in_2.shape

    # Step 1: LayerNorm + GELU -> contiguous temp buffer
    temp = torch.empty_like(in_2)
    _ln_gelu_1d_kernel[(B * M,)](
        in_2, in_1, in_0, temp,
        N,
    )

    # Step 2: Tiled coalesced transpose
    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (triton.cdiv(M, meta['TILE_M']),
                triton.cdiv(N, meta['TILE_N']),
                B)

    _tiled_transpose_kernel[grid](temp, out, B, M, N)

    return out


# ---------------------------------------------------------------------------
# replacement_func - returns the callable (not the result of calling it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_layernorm_transpose_gelu