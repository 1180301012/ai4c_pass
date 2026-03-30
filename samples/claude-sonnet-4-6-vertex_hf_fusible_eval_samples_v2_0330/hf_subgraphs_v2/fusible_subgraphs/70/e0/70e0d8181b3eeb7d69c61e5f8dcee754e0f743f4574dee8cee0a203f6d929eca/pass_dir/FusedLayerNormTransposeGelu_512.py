import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_2 = torch.nn.functional.layer_norm(in_2, (512,), in_1, in_0, 1e-05)
    tmp_3 = tmp_2.transpose(-2, -1)
    tmp_4 = torch.nn.functional.gelu(tmp_3)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ── Strategy ──────────────────────────────────────────────────────────────────
#
# Single fused kernel: LayerNorm + GELU + Transpose.
# Process TILE_N consecutive rows at once.
#
#  1. Load [TILE_N, H] from input  (coalesced along H)
#  2. LayerNorm + GELU per row (2-D reduces over axis=1)
#  3. tl.trans() → [H, TILE_N]
#  4. Explicit 2-D store to out[b, :, n_start:n_start+TILE_N]
#       inner dim = TILE_N (consecutive N values) → coalesced writes
#       tl.store auto-casts fp32 → fp16/bf16  (no block_ptr dtype mismatch)
#  5. NO temp buffer, NO view — output goes directly to [B, H, N].
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'TILE_N': 4,  'BLOCK_H': 512}, num_warps=4),
        triton.Config({'TILE_N': 8,  'BLOCK_H': 512}, num_warps=4),
        triton.Config({'TILE_N': 16, 'BLOCK_H': 512}, num_warps=8),
        triton.Config({'TILE_N': 32, 'BLOCK_H': 512}, num_warps=8),
        triton.Config({'TILE_N': 4,  'BLOCK_H': 512}, num_warps=8),
        triton.Config({'TILE_N': 8,  'BLOCK_H': 512}, num_warps=8),
        triton.Config({'TILE_N': 16, 'BLOCK_H': 512}, num_warps=16),
        triton.Config({'TILE_N': 32, 'BLOCK_H': 512}, num_warps=16),
    ],
    key=['N', 'H'],
)
@triton.jit
def fused_ln_gelu_transpose_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, H, eps,
    TILE_N:  tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    n_start    = pid_n * TILE_N
    tile_n_off = tl.arange(0, TILE_N)   # [TILE_N]
    h_off      = tl.arange(0, BLOCK_H)  # [H]

    n_indices = n_start + tile_n_off     # [TILE_N]
    n_mask    = n_indices < N            # [TILE_N]

    # ── 1. Load [TILE_N, H]  (coalesced along H) ─────────────────────────────
    x = tl.load(
        x_ptr + pid_b * N * H + n_indices[:, None] * H + h_off[None, :],
        mask=n_mask[:, None],
        other=0.0,
    ).to(tl.float32)                     # [TILE_N, H]

    w = tl.load(weight_ptr + h_off).to(tl.float32)  # [H]
    b = tl.load(bias_ptr   + h_off).to(tl.float32)  # [H]

    # ── 2. LayerNorm per row  (reduce over H = axis 1) ───────────────────────
    mean = tl.sum(x, axis=1)[:, None] / H            # [TILE_N, 1]
    xc   = x - mean                                   # [TILE_N, H]
    var  = tl.sum(xc * xc, axis=1)[:, None] / H      # [TILE_N, 1]
    rstd = tl.rsqrt(var + eps)                        # [TILE_N, 1]
    y    = xc * rstd * w[None, :] + b[None, :]       # [TILE_N, H]

    # ── 3. GELU (exact erf form) ──────────────────────────────────────────────
    gelu = 0.5 * y * (1.0 + tl.math.erf(y * 0.7071067811865476))  # [TILE_N, H]

    # ── 4. Transpose [TILE_N, H] → [H, TILE_N] ───────────────────────────────
    gelu_t = tl.trans(gelu)                           # [H, TILE_N]

    # ── 5. Store [H, TILE_N] to out[b, :, n_start:n_start+TILE_N] ───────────
    # out layout [B, H, N]: inner dim TILE_N → consecutive N values → coalesced
    # tl.store auto-casts fp32 → output dtype (fp16 or bf16)
    n_out_off = n_start + tl.arange(0, TILE_N)
    out_ptrs  = out_ptr + pid_b * H * N + h_off[:, None] * N + n_out_off[None, :]
    tl.store(out_ptrs, gelu_t, mask=(n_out_off < N)[None, :])


@torch.fx.wrap
def fused_ln_transpose_gelu(in_0, in_1, in_2):
    """
    in_0 : bias   [H]
    in_1 : weight [H]
    in_2 : input  [B, N, H]
    returns: [B, H, N]  — LN + GELU + Transpose in one fused kernel,
                          output written directly to [B, H, N] (no temp buffer).
    """
    B, N, H = in_2.shape
    out = torch.empty((B, H, N), dtype=in_2.dtype, device=in_2.device)

    fused_ln_gelu_transpose_kernel[
        lambda META: (triton.cdiv(N, META['TILE_N']), B)
    ](in_2, in_1, in_0, out, N, H, 1e-05)

    return out


def replacement_func():
    return fused_ln_transpose_gelu