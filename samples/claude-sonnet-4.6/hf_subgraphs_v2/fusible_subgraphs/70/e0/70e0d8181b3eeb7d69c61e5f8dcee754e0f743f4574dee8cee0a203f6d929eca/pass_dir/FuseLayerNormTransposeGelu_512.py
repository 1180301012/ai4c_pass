import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern – mirrors model.py exactly (no cleanup statements)
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ===========================================================================
# Fused 2-D tiled kernel: layer_norm + transpose(-2,-1) + gelu
#
# Input  x   : [B, N, D]    e.g. [1, 3999, 512]
# Weight w   : [D]
# Bias   b   : [D]
# Output out : [B, D, N]
#
# Grid  : (ceil(N / TILE_N), B)
# Block : TILE_N rows processed per program.
#
# Coalescing: after computing the [TILE_N, D] result we transpose it to
# [D, TILE_N] and write each output row as TILE_N consecutive fp16 values.
#
# Optimisations vs naïve single-row kernel:
#   • BLOCK_D is a compile-time constexpr → divisions become multiplications
#   • tl.math.rsqrt → single SFU instruction (vs 2 for 1/sqrt)
#   • BLOCK_D×TILE_N data loaded / stored in 2-D coalesced pattern
# ===========================================================================
@triton.autotune(
    configs=[
        triton.Config({'TILE_N': 32, 'BLOCK_D': 512}, num_warps=4),
        triton.Config({'TILE_N': 32, 'BLOCK_D': 512}, num_warps=8),
        triton.Config({'TILE_N': 32, 'BLOCK_D': 512}, num_warps=16),
        triton.Config({'TILE_N': 16, 'BLOCK_D': 512}, num_warps=4),
        triton.Config({'TILE_N': 16, 'BLOCK_D': 512}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def _ln_tp_gelu_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    N,
    D,
    eps,
    TILE_N:  tl.constexpr,   # rows per program   (e.g. 32)
    BLOCK_D: tl.constexpr,   # = D                (e.g. 512)
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start  = pid_n * TILE_N
    t_offs   = tl.arange(0, TILE_N)    # [TILE_N]
    d_offs   = tl.arange(0, BLOCK_D)   # [BLOCK_D]
    n_offs   = n_start + t_offs         # [TILE_N]
    row_mask = n_offs < N               # [TILE_N]

    in_base  = pid_b * N * BLOCK_D
    out_base = pid_b * BLOCK_D * N

    # ---- load [TILE_N, BLOCK_D] tile (coalesced read) ----
    x = tl.load(x_ptr + in_base + n_offs[:, None] * BLOCK_D + d_offs[None, :],
                mask=row_mask[:, None], other=0.0).to(tl.float32)

    # ---- layer-norm: mean + rstd (BLOCK_D is constexpr → compile-time division) ----
    inv_D = 1.0 / BLOCK_D                             # compile-time constant
    mean  = tl.sum(x, axis=1)[:, None] * inv_D        # [TILE_N, 1]
    x_c   = x - mean                                   # [TILE_N, BLOCK_D]
    var   = tl.sum(x_c * x_c, axis=1)[:, None] * inv_D  # [TILE_N, 1]
    rstd  = tl.math.rsqrt(var + eps)                   # [TILE_N, 1]  (single SFU)
    x_n   = x_c * rstd                                 # [TILE_N, BLOCK_D]

    # ---- affine ----
    w     = tl.load(w_ptr + d_offs).to(tl.float32)    # [BLOCK_D]
    b_v   = tl.load(b_ptr + d_offs).to(tl.float32)    # [BLOCK_D]
    y     = x_n * w[None, :] + b_v[None, :]           # [TILE_N, BLOCK_D]

    # ---- GELU (exact) ----
    y_out = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    # ---- transposed store [BLOCK_D, TILE_N]: TILE_N consecutive fp16 per row ----
    out_offs = d_offs[:, None] * N + n_offs[None, :]   # [BLOCK_D, TILE_N]
    tl.store(out_ptr + out_base + out_offs,
             tl.trans(y_out),
             mask=row_mask[None, :])


# ===========================================================================
# Wrapper
# ===========================================================================
@torch.fx.wrap
def _fused_ln_tp_gelu(bias, weight, x):
    B = x.shape[0]
    N = x.shape[1]
    D = x.shape[2]

    out  = torch.empty((B, D, N), dtype=x.dtype, device=x.device)
    grid = lambda meta: (triton.cdiv(N, meta['TILE_N']), B)

    _ln_tp_gelu_kernel[grid](
        x, weight, bias, out,
        N, D, 1e-5,
    )

    return out


def replacement_func():
    return _fused_ln_tp_gelu