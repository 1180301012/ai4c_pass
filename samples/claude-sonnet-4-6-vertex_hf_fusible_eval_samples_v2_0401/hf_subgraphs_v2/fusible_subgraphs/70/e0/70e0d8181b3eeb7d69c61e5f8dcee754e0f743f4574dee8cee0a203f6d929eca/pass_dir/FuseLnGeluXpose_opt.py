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
# Fused kernel: LayerNorm + GELU + coalesced tiled transpose [B,M,N]->[B,N,M]
#
# Design: 2-D tile [TILE_M rows x BLOCK_N cols] per CTA.
#   Reads  [TILE_M, BLOCK_N]: stride-1 in N -> coalesced
#   LayerNorm per row with axis=1 reduction (float32 accumulation)
#   GELU (exact erf form)
#   Writes [BLOCK_N, TILE_M] via tl.trans: stride-1 in M -> coalesced
#
# Key autotune insight: TILE_M == num_warps -> 1 row per warp ->
#   pure warp-level reduction (no cross-warp barrier), lowest overhead.
# N is always 512 in this pattern, so we use 1/512 as a compile-time const.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # --- 1 row per warp (TILE_M == num_warps) ---
        # TILE_M=4,  4 warps -> 128 threads, 1000 CTAs ~ 1.1 waves (BEST occupancy)
        triton.Config({'TILE_M':  4, 'BLOCK_N': 512}, num_warps= 4),
        # TILE_M=8,  8 warps -> 256 threads, 500 CTAs
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 8),
        # TILE_M=16, 16 warps-> 512 threads, 250 CTAs
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps=16),
        # TILE_M=32, 32 warps->1024 threads, 125 CTAs ~ 1.1 waves
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=32),
        # --- 2 rows per warp (more rows per CTA, better weight/bias reuse) ---
        triton.Config({'TILE_M':  4, 'BLOCK_N': 512}, num_warps= 2),
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=16),
        # --- 4 rows per warp ---
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 2),
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=16),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_ln_gelu_xpose(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    M,
    N,
    TILE_M:  tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    input  : [B, M, N]  contiguous
    output : [B, N, M]  contiguous  (layernorm -> GELU -> transposed)
    N is always 512 for this pattern; we use compile-time reciprocal.
    """
    INV_N = 1.0 / 512.0     # compile-time constant (N always 512 in pattern)

    pid = tl.program_id(0)
    num_tiles_m = tl.cdiv(M, TILE_M)
    pid_b = pid // num_tiles_m
    pid_m = pid %  num_tiles_m

    m_base = pid_m * TILE_M
    m_offs = m_base + tl.arange(0, TILE_M)   # [TILE_M]
    n_offs = tl.arange(0, BLOCK_N)            # [BLOCK_N]

    m_mask = m_offs < M
    n_mask = n_offs < N

    # ------------------------------------------------------------------
    # 1. Coalesced load [TILE_M, BLOCK_N] (stride-1 in N)
    # ------------------------------------------------------------------
    in_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(input_ptr + in_offs,
                 mask=m_mask[:, None] & n_mask[None, :],
                 other=0.0).to(tl.float32)              # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 2. LayerNorm (axis=1 reduction; INV_N avoids runtime division)
    # ------------------------------------------------------------------
    mean  = tl.sum(x, axis=1) * INV_N                   # [TILE_M]
    diff  = x - mean[:, None]                            # [TILE_M, BLOCK_N]
    var   = tl.sum(diff * diff, axis=1) * INV_N          # [TILE_M]
    rstd  = 1.0 / tl.sqrt(var + 1e-5)                   # [TILE_M]
    x_norm = diff * rstd[:, None]                        # [TILE_M, BLOCK_N]

    weight = tl.load(weight_ptr + n_offs,
                     mask=n_mask, other=1.0).to(tl.float32)  # [BLOCK_N]
    bias   = tl.load(bias_ptr   + n_offs,
                     mask=n_mask, other=0.0).to(tl.float32)  # [BLOCK_N]

    y = x_norm * weight[None, :] + bias[None, :]        # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 3. Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    # ------------------------------------------------------------------
    INV_SQRT2 = 0.7071067811865476
    y_gelu = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))

    # ------------------------------------------------------------------
    # 4. Coalesced write [BLOCK_N, TILE_M] via tl.trans (stride-1 in M)
    # ------------------------------------------------------------------
    out_offs = pid_b * N * M + n_offs[:, None] * M + m_offs[None, :]
    tl.store(output_ptr + out_offs,
             tl.trans(y_gelu),
             mask=n_mask[:, None] & m_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_layernorm_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [N]          float16/bfloat16
    in_1 : weight [N]          float16/bfloat16
    in_2 : input  [B, M, N]    float16/bfloat16
    Returns [B, N, M]: layer_norm -> transpose(-2,-1) -> gelu
    """
    B, M, N = in_2.shape
    out = torch.empty((B, N, M), dtype=in_2.dtype, device=in_2.device)

    def grid(meta):
        return (triton.cdiv(B * M, meta['TILE_M']),)

    _fused_ln_gelu_xpose[grid](
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