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
# Key design: 2-D tiling over [TILE_M rows, BLOCK_N cols]
#   - Reads  [TILE_M, BLOCK_N] from input   (stride-1 in N -> coalesced)
#   - Runs LayerNorm (axis=1 reduction) + GELU per row in float32
#   - Writes [BLOCK_N, TILE_M] via tl.trans (stride-1 in M -> coalesced)
#
# Register pressure analysis for TILE_M=32, BLOCK_N=512, num_warps=16
#   (512 threads):  32*512/512 = 32 elements per thread.
#   With register reuse (x->diff->x_norm->y->y_gelu) + weight/bias/rstd:
#   ~42 live regs/thread -> 3 CTAs/SM -> 125 CTAs all run in ONE WAVE.
#
# Autotune explores TILE_M in {8,16,32,64} x num_warps in {4,8,16,32}
# to let the framework pick the best occupancy/ILP trade-off.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # TILE_M=8: 500 CTAs, each CTA small -> high concurrency
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M':  8, 'BLOCK_N': 512}, num_warps= 8),
        # TILE_M=16: 250 CTAs
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 16, 'BLOCK_N': 512}, num_warps=16),
        # TILE_M=32: 125 CTAs, 3 CTAs/SM with num_warps=16 -> 1 wave
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps= 4),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 32, 'BLOCK_N': 512}, num_warps=16),
        # TILE_M=64: 63 CTAs, ~1 CTA/SM -> minimal overhead
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps= 8),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=16),
        triton.Config({'TILE_M': 64, 'BLOCK_N': 512}, num_warps=32),
    ],
    key=['M', 'N'],
)
@triton.jit
def _fused_ln_gelu_transpose_2d(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    B,
    M,
    N,
    TILE_M:  tl.constexpr,   # rows per CTA (autotuned)
    BLOCK_N: tl.constexpr,   # must be >= N and a power of 2
):
    """
    input  : [B, M, N]  contiguous
    output : [B, N, M]  contiguous (layernorm -> GELU -> transposed)
    """
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
    # 1. Coalesced read [TILE_M, BLOCK_N] from input (stride-1 in N)
    # ------------------------------------------------------------------
    in_offs = pid_b * M * N + m_offs[:, None] * N + n_offs[None, :]
    x = tl.load(input_ptr + in_offs,
                 mask=m_mask[:, None] & n_mask[None, :],
                 other=0.0).to(tl.float32)              # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 2. LayerNorm – row-wise reduction (axis=1 = N dimension)
    # ------------------------------------------------------------------
    mean  = tl.sum(x, axis=1) / N                       # [TILE_M]
    diff  = x - mean[:, None]                            # [TILE_M, BLOCK_N]
    var   = tl.sum(diff * diff, axis=1) / N             # [TILE_M]
    rstd  = 1.0 / tl.sqrt(var + 1e-5)                  # [TILE_M]
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
    y_gelu = 0.5 * y * (1.0 + tl.math.erf(y * INV_SQRT2))  # [TILE_M, BLOCK_N]

    # ------------------------------------------------------------------
    # 4. Coalesced write [BLOCK_N, TILE_M] via tl.trans (stride-1 in M)
    # ------------------------------------------------------------------
    out_offs = pid_b * N * M + n_offs[:, None] * M + m_offs[None, :]
    tl.store(output_ptr + out_offs,
             tl.trans(y_gelu),                           # [BLOCK_N, TILE_M]
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

    _fused_ln_gelu_transpose_2d[grid](
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