import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: LayerNorm -> transpose(-2,-1) -> GELU
# Input shapes: in_2=[B,T,C], in_1=weight[C], in_0=bias[C]
# Output shape: [B,C,T]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    # in_0=bias[C], in_1=weight[C], in_2=input[B,T,C]
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused LayerNorm + Transpose + GELU  (2D tiled, coalesced I/O)
#
# Key insight: process BLOCK_T consecutive rows per program.
#   Reads  x[b, t_start:t_start+BLOCK_T, :]  → [BLOCK_T, BLOCK_C] (coalesced)
#   Computes LayerNorm per row (axis=1 reduction) then GELU
#   Stores  out[b, :, t_start:t_start+BLOCK_T] → [BLOCK_C, BLOCK_T] via tl.trans
#     → consecutive t-positions are consecutive in memory → coalesced writes!
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 16,  'BLOCK_C': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_T': 16,  'BLOCK_C': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_T': 32,  'BLOCK_C': 512}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_T': 32,  'BLOCK_C': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_T': 32,  'BLOCK_C': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_T': 64,  'BLOCK_C': 512}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_T': 64,  'BLOCK_C': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_C': 512}, num_warps=16, num_stages=2),
    ],
    key=['T', 'C'],
)
@triton.jit
def _fused_ln_tgelu_2d_kernel(
    x_ptr,    # [B, T, C] – input
    w_ptr,    # [C]       – LN weight (gamma)
    b_ptr,    # [C]       – LN bias   (beta)
    out_ptr,  # [B, C, T] – output
    B, T, C,
    eps,
    BLOCK_T: tl.constexpr,   # rows processed per program
    BLOCK_C: tl.constexpr,   # = C = 512
):
    pid         = tl.program_id(0)
    num_t_tiles = tl.cdiv(T, BLOCK_T)
    b           = pid // num_t_tiles
    t_tile      = pid  % num_t_tiles
    t_start     = t_tile * BLOCK_T

    t_offs = t_start + tl.arange(0, BLOCK_T)   # [BLOCK_T]
    c_offs = tl.arange(0, BLOCK_C)             # [BLOCK_C]
    t_mask = t_offs < T                        # boundary guard

    # ---- Load x[b, t_start:, :] → [BLOCK_T, BLOCK_C]  (coalesced reads) ----
    x_ptrs = x_ptr + b * T * C + t_offs[:, None] * C + c_offs[None, :]
    x_raw  = tl.load(x_ptrs, mask=t_mask[:, None], other=0.0)
    x      = x_raw.to(tl.float32)              # [BLOCK_T, BLOCK_C]

    # ---- LayerNorm over C dimension (axis=1) ----
    mean  = tl.sum(x, axis=1)[:, None] / C            # [BLOCK_T, 1]
    x_c   = x - mean                                   # [BLOCK_T, BLOCK_C]
    var   = tl.sum(x_c * x_c, axis=1)[:, None] / C    # [BLOCK_T, 1]
    rstd  = tl.rsqrt(var + eps)                        # [BLOCK_T, 1]
    x_hat = x_c * rstd                                 # [BLOCK_T, BLOCK_C]

    # ---- Affine transform ----
    w     = tl.load(w_ptr + c_offs).to(tl.float32)    # [BLOCK_C]
    b_v   = tl.load(b_ptr + c_offs).to(tl.float32)    # [BLOCK_C]
    y     = x_hat * w[None, :] + b_v[None, :]         # [BLOCK_T, BLOCK_C]

    # ---- GELU (exact: x*0.5*(1+erf(x/sqrt(2)))) ----
    y_g   = y * 0.5 * (1.0 + tl.math.erf(y * 0.7071067811865476))

    # ---- Cast back to original dtype ----
    y_out = y_g.to(x_raw.dtype)                        # [BLOCK_T, BLOCK_C]

    # ---- Store transposed: out[b, :, t_start:] → [BLOCK_C, BLOCK_T] ----
    # out_ptrs[c, t] = out_ptr + b*C*T + c*T + (t_start+dt)  → shape [BLOCK_C, BLOCK_T]
    # tl.trans(y_out) re-orders [BLOCK_T, BLOCK_C] → [BLOCK_C, BLOCK_T]
    # Warp assigns threads along the T dimension first → coalesced writes!
    out_ptrs = out_ptr + b * C * T + c_offs[:, None] * T + t_offs[None, :]
    out_mask = t_mask[None, :]                          # [1, BLOCK_T] broadcasts
    tl.store(out_ptrs, tl.trans(y_out), mask=out_mask)


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_ln_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [C]
    in_1 : weight [C]
    in_2 : input  [B, T, C]
    returns [B, C, T]  –  LayerNorm -> transpose(-2,-1) -> GELU
    """
    B = in_2.shape[0]
    T = in_2.shape[1]
    C = in_2.shape[2]

    out = torch.empty((B, C, T), dtype=in_2.dtype, device=in_2.device)

    grid = lambda meta: (B * triton.cdiv(T, meta['BLOCK_T']),)
    _fused_ln_tgelu_2d_kernel[grid](
        in_2, in_1, in_0, out,
        B, T, C,
        1e-5,
    )
    return out


def replacement_func():
    return fused_ln_transpose_gelu