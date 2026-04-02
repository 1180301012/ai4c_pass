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
# Fused kernel: LayerNorm + GELU + coalesced tiled transpose
#
# 2-D tile [TILE_M rows x BLOCK_N=512 cols]:
#   Reads  [TILE_M, 512] – stride-1 in N -> coalesced
#   LayerNorm per row (axis=1 reduction in fp32)
#   Exact GELU (erf form)
#   Writes [512, TILE_M] via tl.trans – stride-1 in M -> coalesced
#
# Micro-optimisations vs previous version:
#   - N-direction mask removed (BLOCK_N == N == 512 always, so mask is all-True)
#   - INV_N = 1/512 is a compile-time constant (avoids runtime division)
#   - TILE_M=2 added: 2000 CTAs -> ~32 CTAs/SM -> 100% thread occupancy
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # TILE_M = num_warps -> 1 row per warp -> pure warp-level reduction
        triton.Config({'TILE_M':  2, 'BLOCK_N': 512}, num_warps= 2),  # 2000 CTAs
        triton.Config({'TILE_M':  4, 'BLOCK_N': 512}, num_warps= 4),  # 1000 CTAs
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 8),  #  500 CTAs
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps=16),  #  250 CTAs
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=32),  #  125 CTAs
        # 2 rows per warp (more ILP per CTA, better weight/bias reuse)
        triton.Config({'TILE_M':  4, 'BLOCK_N': 512}, num_warps= 2),
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=16),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=32),
        # 4 rows per warp
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def _kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    M,
    N,
    TILE_M:  tl.constexpr,
    BLOCK_N: tl.constexpr,   # always 512 == N
):
    # N is always 512 in this pattern; reciprocal is a compile-time const
    INV_N    = 1.0 / 512.0
    INV_SQRT2 = 0.7071067811865476

    pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(M, TILE_M)
    pid_b = pid // num_tiles_m
    pid_m = pid %  num_tiles_m

    m_base = pid_m * TILE_M
    m_offs = m_base + tl.arange(0, TILE_M)   # [TILE_M]
    n_offs = tl.arange(0, BLOCK_N)            # [BLOCK_N]  (== [0..511])

    m_mask = m_offs < M    # [TILE_M]  (only needed for M boundary)
    # n_mask is always True since BLOCK_N == N == 512 -- skipped

    # ------------------------------------------------------------------
    # 1. Load [TILE_M, BLOCK_N] – no N-mask (always valid)
    # ------------------------------------------------------------------
    in_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(input_ptr + in_offs,
                 mask=m_mask[:, None],
                 other=0.0).to(tl.float32)              # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 2. LayerNorm (axis=1 = N direction)
    # ------------------------------------------------------------------
    mean  = tl.sum(x, axis=1) * INV_N                   # [TILE_M]
    diff  = x - mean[:, None]                            # [TILE_M, BLOCK_N]
    var   = tl.sum(diff * diff, axis=1) * INV_N          # [TILE_M]
    rstd  = 1.0 / tl.sqrt(var + 1e-5)                   # [TILE_M]
    x_norm = diff * rstd[:, None]                        # [TILE_M, BLOCK_N]

    # Load weight / bias (no N-mask; always 512 elements)
    weight = tl.load(weight_ptr + n_offs).to(tl.float32) # [BLOCK_N]
    bias   = tl.load(bias_ptr   + n_offs).to(tl.float32) # [BLOCK_N]

    y = x_norm * weight[None, :] + bias[None, :]        # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 3. Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # ------------------------------------------------------------------
    y_gelu = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    # ------------------------------------------------------------------
    # 4. Transpose + store (no N-mask)
    # ------------------------------------------------------------------
    out_offs = pid_b * N * M + n_offs[:, None] * M + m_offs[None, :]
    tl.store(output_ptr + out_offs,
             tl.trans(y_gelu),                           # [BLOCK_N, TILE_M]
             mask=m_mask[None, :])


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
    """
    B, M, N = in_2.shape
    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (triton.cdiv(B * M, meta['TILE_M']),)

    _kernel[grid](
        in_2,   # input_ptr
        in_1,   # weight_ptr
        in_0,   # bias_ptr
        out,
        B, M, N,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func - returns the callable (not the result of calling it)
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_layernorm_transpose_gelu