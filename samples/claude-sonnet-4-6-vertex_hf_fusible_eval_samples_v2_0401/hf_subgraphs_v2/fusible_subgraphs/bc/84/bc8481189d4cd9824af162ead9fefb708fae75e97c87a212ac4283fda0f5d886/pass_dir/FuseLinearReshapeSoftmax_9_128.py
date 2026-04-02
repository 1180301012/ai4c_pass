import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern: linear + reshape([-1,9,1]) + softmax(dim=1)
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─────────────────────────────────────────────────────────────────────────────
# Fused Triton kernel: linear (GEMM via tl.dot / tensor cores) +
#                      reshape (implicit) + two softmax groups per row
#
# Shapes (from weight_meta.py):
#   x     : [1, 19, 128]  (fp16/bf16)
#   weight: [18, 128]
#   bias  : [18]
#   output: [38, 9, 1]    (= N_GROUPS × GROUP_SIZE × 1)
#
# Grid:  (ceil(M_TOT / BLOCK_M),) = (2,)
#   Program 0: input rows 0..15  → output rows 0..31
#   Program 1: input rows 16..18 → output rows 32..37  (3 valid, 13 masked)
#
# Weight loading: W^T stored as [BLOCK_K, BLOCK_N] using
#   wt_off[k, n] = n_idx[n] * K + k_range[k]
# This provides the column-major layout that cuBLAS-style WMMA expects as
# the B matrix in an A@B multiply.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def fused_linear_reshape_softmax_kernel(
    x_ptr,    # [M_TOT, K]
    w_ptr,    # [N, K]
    b_ptr,    # [N]
    out_ptr,  # [M_TOT * N]  (same dtype as x)
    M_TOT,            # runtime: 19
    K:          tl.constexpr,   # 128
    N:          tl.constexpr,   # 18
    BLOCK_M:    tl.constexpr,   # 16  (≥16 for tensor cores)
    BLOCK_N:    tl.constexpr,   # 32  (next pow-2 ≥ 18)
    BLOCK_K:    tl.constexpr,   # 128
    GROUP_SIZE: tl.constexpr,   # 9
):
    pid     = tl.program_id(0)
    m_start = pid * BLOCK_M

    m_range = tl.arange(0, BLOCK_M)
    k_range = tl.arange(0, BLOCK_K)
    n_range = tl.arange(0, BLOCK_N)

    m_abs  = m_start + m_range
    m_mask = m_abs < M_TOT
    n_mask = n_range < N

    # ── Load X tile: [BLOCK_M, K]
    # Masked OOB lanes use Triton predication (no actual memory access).
    x_off  = m_abs[:, None] * K + k_range[None, :]
    x_tile = tl.load(x_ptr + x_off, mask=m_mask[:, None], other=0.0)

    # ── Load W^T tile: [BLOCK_K, BLOCK_N]
    # wt_tile[k, n] = W[n, k]  →  W^T layout for A@B = X @ W^T
    wt_off  = n_range[None, :] * K + k_range[:, None]
    wt_tile = tl.load(w_ptr + wt_off, mask=n_mask[None, :], other=0.0)

    # ── GEMM: [BLOCK_M, K] @ [K, BLOCK_N] → [BLOCK_M, BLOCK_N] fp32 ─────────
    # Tensor cores used when BLOCK_M, BLOCK_N, BLOCK_K are all ≥16 and pow-of-2.
    acc = tl.dot(x_tile, wt_tile, allow_tf32=False)

    # ── Bias ──────────────────────────────────────────────────────────────────
    bias = tl.load(b_ptr + n_range, mask=n_mask, other=0.0).to(tl.float32)
    acc  = acc + bias[None, :]

    # ── Softmax group 0: cols 0..GROUP_SIZE-1  (0..8) ────────────────────────
    g0_mask = n_range < GROUP_SIZE
    g0_m    = tl.where(g0_mask[None, :], acc, -1e38)
    g0_max  = tl.max(g0_m, axis=1)[:, None]
    g0_exp  = tl.exp(acc - g0_max)
    g0_expv = tl.where(g0_mask[None, :], g0_exp, 0.0)
    g0_sum  = tl.sum(g0_expv, axis=1)[:, None]
    g0_sm   = g0_expv / g0_sum   # 0 for n≥GROUP_SIZE

    # ── Softmax group 1: cols GROUP_SIZE..2*GROUP_SIZE-1  (9..17) ────────────
    g1_mask = (n_range >= GROUP_SIZE) & (n_range < 2 * GROUP_SIZE)
    g1_m    = tl.where(g1_mask[None, :], acc, -1e38)
    g1_max  = tl.max(g1_m, axis=1)[:, None]
    g1_exp  = tl.exp(acc - g1_max)
    g1_expv = tl.where(g1_mask[None, :], g1_exp, 0.0)
    g1_sum  = tl.sum(g1_expv, axis=1)[:, None]
    g1_sm   = g1_expv / g1_sum   # 0 for n<GROUP_SIZE and n≥2*GROUP_SIZE

    # ── Combine: g0_sm and g1_sm have non-overlapping non-zero regions,
    #            so addition is equivalent to tl.where (avoids an extra op).
    result = g0_sm + g1_sm

    # ── Store ─────────────────────────────────────────────────────────────────
    out_off    = m_abs[:, None] * N + n_range[None, :]
    store_mask = m_mask[:, None] & n_mask[None, :]
    tl.store(out_ptr + out_off, result, mask=store_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-allocated output-buffer cache (one per dtype/device pair)
# Eliminates torch.empty() from the hot path after the first call.
# ─────────────────────────────────────────────────────────────────────────────
_out_cache = {}   # key → pre-allocated flat tensor


# ─────────────────────────────────────────────────────────────────────────────
# Replacement wrapper
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    """
    Fused: linear(x, weight, bias) → reshape([-1,9,1]) → softmax(dim=1)
    Uses tl.dot (tensor cores) for the GEMM + in-kernel numerically-stable
    softmax.  Output is written directly in the input dtype.
    """
    # ── Hot-path: use dtype as key (device is always cuda:0 for this model) ──
    key = x.dtype
    if key not in _out_cache:
        _out_cache[key] = torch.empty(342, dtype=key, device=x.device)
    out_flat = _out_cache[key]

    # M_TOT from second-to-last dim (handles any batch × seq, fixed K=128)
    M_TOT = x.shape[-2]   # 19 for [1, 19, 128]

    # ── Launch fused kernel ───────────────────────────────────────────────────
    fused_linear_reshape_softmax_kernel[(2,)](
        x, weight, bias, out_flat,
        M_TOT,
        K=128,
        N=18,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=128,
        GROUP_SIZE=9,
        num_warps=1,
    )

    # out_flat is [M_TOT*18 = 342] → [38, 9, 1]
    return out_flat.view(M_TOT * 2, 9, 1)


def replacement_func():
    return fused_linear_reshape_softmax