import torch
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = in_0 * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


def replacement_args(in_0):
    return (in_0,)


# Fused scale + softmax + transpose.
#
# KEY DESIGN: BLOCK_M=16, N=400=25×16 — every tile is a FULL TILE.
# 400 divides evenly by 16, so rows 0..399 never touch a partial last tile.
# This eliminates the OOB/NaN correctness bug that occurs with BLOCK_M >= 32.
#
# Both loads and stores are coalesced:
#   Load  [16, 512] order=(1,0): j-dim innermost → consecutive j → coalesced reads
#   Store [512, 16] order=(1,0): i-dim (16) innermost → coalesced writes
@triton.jit
def _fused_scale_softmax_transpose_kernel(
    input_ptr,
    output_ptr,
    H,             # grid helper — not used in offset arithmetic
    scale,
    BLOCK_M: tl.constexpr,   # 16 rows per program  (400 = 25×16, no partial tiles)
    BLOCK_N: tl.constexpr,   # 512 >= N=400, power-of-2 (no partial last tile in j)
    N: tl.constexpr,         # 400
):
    pid   = tl.program_id(0)
    num_m = tl.cdiv(N, BLOCK_M)   # = 25 (compile-time constant, no partial tiles)
    bh    = pid // num_m
    m_blk = pid  % num_m

    rows = m_blk * BLOCK_M + tl.arange(0, BLOCK_M)   # [16]  — always within [0, N)
    cols = tl.arange(0, BLOCK_N)                       # [512]
    mask_n = cols < N                                  # [512] True for j < 400

    # ------------------------------------------------------------------
    # Load [16, 512] tile.  No row masking needed (full tiles guaranteed).
    # ------------------------------------------------------------------
    base_in = bh * N * N
    in_offs = base_in + rows[:, None] * N + cols[None, :]   # [16, 512]
    x = tl.load(
        input_ptr + in_offs,
        mask=mask_n[None, :],
        other=-float('inf'),
    )
    x = x.to(tl.float32) * scale

    # ------------------------------------------------------------------
    # Numerically-stable softmax along the N axis (axis=1)
    # ------------------------------------------------------------------
    x_max = tl.max(x, axis=1)               # [16]
    x     = x - x_max[:, None]
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=1)           # [16]
    x     = x_exp / x_sum[:, None]          # [16, 512]

    # Cast back to original dtype
    x = x.to(input_ptr.dtype.element_ty)    # [16, 512]

    # ------------------------------------------------------------------
    # Transpose [16, 512] → [512, 16] and store coalesced.
    # ------------------------------------------------------------------
    x_T = tl.trans(x)   # [512, 16]

    out_offs = base_in + cols[:, None] * N + rows[None, :]  # [512, 16]
    tl.store(
        output_ptr + out_offs,
        x_T,
        mask=mask_n[:, None],   # col mask only; rows always valid
    )


@torch.fx.wrap
def fused_scale_softmax_transpose(in_0):
    B, H, N, N2 = in_0.shape
    out = torch.empty((B, H, N, N2), dtype=in_0.dtype, device=in_0.device)

    # BLOCK_M=16: 400 = 25×16 — zero partial tiles, no OOB issues.
    # num_warps=8 (256 threads) → 8×512/256 = 16 elements/thread → good occupancy.
    BLOCK_M = 16
    grid = (B * H * N // BLOCK_M,)   # exact division, no ceiling needed

    _fused_scale_softmax_transpose_kernel[grid](
        in_0,
        out,
        H,
        0.1767766952966369,
        BLOCK_M=BLOCK_M,
        BLOCK_N=512,
        N=N,
        num_warps=8,
    )

    return out


def replacement_func():
    return fused_scale_softmax_transpose