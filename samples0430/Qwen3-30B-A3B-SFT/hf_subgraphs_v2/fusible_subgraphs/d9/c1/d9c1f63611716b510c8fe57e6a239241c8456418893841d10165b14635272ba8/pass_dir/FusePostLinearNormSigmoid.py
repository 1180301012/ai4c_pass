import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: entire forward pass including the linear GEMM
# ---------------------------------------------------------------------------
def pattern(in_8, in_7, in_6, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0):
    linear = torch.nn.functional.linear(in_8, in_7, in_6)
    tmp_9 = torch.nn.functional.layer_norm(linear, (256,), in_3, in_2, 1e-05)
    tmp_10 = in_9.sigmoid()
    tmp_11 = tmp_9.sigmoid()
    tmp_12 = torch.nn.functional.layer_norm(in_11, (256,), in_5, in_4, 1e-05)
    tmp_13 = torch.nn.functional.layer_norm(in_10, (256,), in_1, in_0, 1e-05)
    tmp_14 = tmp_12.unsqueeze(-2)
    tmp_15 = tmp_11 * tmp_14
    tmp_16 = tmp_10 * tmp_13
    tmp_17 = tmp_15 + tmp_16
    return tmp_17


def replacement_args(in_8, in_7, in_6, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0):
    return (in_8, in_7, in_6, in_3, in_2, in_9, in_11, in_5, in_4, in_10, in_1, in_0)


# ---------------------------------------------------------------------------
# Fused kernel: GEMV (via tl.dot/tensor-cores) + LayerNorm + Sigmoid + combine
#
# Grid: (N_ROWS,)  – one program per outer row (300 rows)
#
# Each program:
#   1. GEMV:  acc[16, N_COLS] = ones_row16[BLOCK_M, BLOCK_K] @ in7.T[BLOCK_K, N_COLS]
#              → only row-0 of acc is valid; use mask to extract
#   2. LN+sigmoid for linear output, in9, in11, in10
#   3. out = s3*ln11 + s9*ln10
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 32},  num_warps=2),
        triton.Config({'BLOCK_K': 32},  num_warps=4),
        triton.Config({'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 32},  num_warps=8),
        triton.Config({'BLOCK_K': 64},  num_warps=8),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_K': 32},  num_warps=16),
        triton.Config({'BLOCK_K': 64},  num_warps=16),
        triton.Config({'BLOCK_K': 128}, num_warps=16),
    ],
    key=['N_COLS', 'DTYPE_STR'],
)
@triton.jit
def fused_linear_ln_sig_kernel(
    in8_ptr,    # [N_ROWS, K]
    in7_ptr,    # [N_COLS, K]  weight
    in6_ptr,    # [N_COLS]     bias
    in9_ptr,    # [N_ROWS, K]
    in11_ptr,   # [N_ROWS, N_COLS]
    in10_ptr,   # [N_ROWS, K]
    w1_ptr, h1_ptr,
    w2_ptr, h2_ptr,
    w3_ptr, h3_ptr,
    out_ptr,    # [N_ROWS, K]
    K: tl.constexpr,          # 256 – constexpr for static loop unrolling
    N_COLS: tl.constexpr,     # 256
    EPS: tl.constexpr,
    DTYPE_STR: tl.constexpr,
    BLOCK_K: tl.constexpr,    # tuned
):
    r      = tl.program_id(0)
    n_cols = tl.arange(0, N_COLS)   # [256]
    k_offs = tl.arange(0, BLOCK_K)  # [BLOCK_K]

    # ── GEMV: acc[N_COLS] += x7_T[BLOCK_K, N_COLS] @ x8[BLOCK_K] ───────────
    # Load in7 transposed as [BLOCK_K, N_COLS]: fast dim = N_COLS → stride-1 coalesced
    acc = tl.zeros((N_COLS,), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        # Load in7.T as [BLOCK_K, N_COLS]:
        #   address[k, n] = in7[n, k] = in7_ptr + n*K + k  → stride-1 in n ✓
        x7_T = tl.load(
            in7_ptr + (k_off + k_offs[:, None]) + n_cols[None, :] * K,
        )  # [BLOCK_K, N_COLS]

        # Load in8 slice  [BLOCK_K]
        x8 = tl.load(in8_ptr + r * K + k_off + k_offs)   # [BLOCK_K]

        # Outer product: acc += sum over k dim → [N_COLS]
        # x8[:, None] * x7_T: [BLOCK_K,1] * [BLOCK_K,N_COLS] → [BLOCK_K,N_COLS]
        # tl.sum(..., axis=0): reduce k dim → [N_COLS]
        acc += tl.sum(x8[:, None] * x7_T, axis=0)         # [N_COLS]

    # Add bias  [N_COLS]
    bias = tl.load(in6_ptr + n_cols)
    linear_col = acc + bias.to(tl.float32)

    # ── LayerNorm(linear_col, w3, h3) ──────────────────────────────────────────
    mean3  = tl.sum(linear_col, 0) / N_COLS
    d3     = linear_col - mean3
    var3   = tl.sum(d3 * d3, 0) / N_COLS
    rstd3  = 1.0 / tl.sqrt(var3 + EPS)
    ln3    = d3 * rstd3 * tl.load(w3_ptr + n_cols).to(tl.float32) + tl.load(h3_ptr).to(tl.float32)
    s3     = 1.0 / (1.0 + tl.exp(-ln3))

    # ── sigmoid(in_9[r, :]) ────────────────────────────────────────────────────
    x9 = tl.load(in9_ptr + r * K + n_cols).to(tl.float32)
    s9 = 1.0 / (1.0 + tl.exp(-x9))

    # ── Load in_11 FIRST (before any in11 write) for layer_norm ───────────────
    x11    = tl.load(in11_ptr + r * N_COLS + n_cols).to(tl.float32)
    mean11 = tl.sum(x11, 0) / N_COLS
    d11    = x11 - mean11
    var11  = tl.sum(d11 * d11, 0) / N_COLS
    rstd11 = 1.0 / tl.sqrt(var11 + EPS)
    ln11   = d11 * rstd11 * tl.load(w1_ptr + n_cols).to(tl.float32) + tl.load(h1_ptr).to(tl.float32)

    # ── LayerNorm(in_10[r, :]) ─────────────────────────────────────────────────
    x10    = tl.load(in10_ptr + r * K + n_cols).to(tl.float32)
    mean10 = tl.sum(x10, 0) / N_COLS
    d10    = x10 - mean10
    var10  = tl.sum(d10 * d10, 0) / N_COLS
    rstd10 = 1.0 / tl.sqrt(var10 + EPS)
    ln10   = d10 * rstd10 * tl.load(w2_ptr + n_cols).to(tl.float32) + tl.load(h2_ptr).to(tl.float32)

    # ── tmp_17 = s3 * ln11 + s9 * ln10 ────────────────────────────────────────
    result = s3 * ln11 + s9 * ln10

    # ── store ──────────────────────────────────────────────────────────────────
    out_off = r * N_COLS + n_cols
    if DTYPE_STR == "float16":
        tl.store(out_ptr + out_off, result.to(tl.float16))
    elif DTYPE_STR == "bfloat16":
        tl.store(out_ptr + out_off, result.to(tl.bfloat16))
    else:
        tl.store(out_ptr + out_off, result)


@torch.fx.wrap
def fused_linear_ln_sig(in_8, in_7, in_6, w3, h3, in_9, in_11, w1, h1, in_10, w2, h2):
    # in_8 : [N_ROWS, 1, K]
    # in_7 : [N_COLS, K]   weight
    # in_6 : [N_COLS]      bias
    # w3,h3  : [N_COLS]    LN params for linear output
    # in_9   : [N_ROWS, 1, K]
    # in_11  : [N_ROWS, N_COLS]
    # w1,h1  : [N_COLS]    LN params for in_11
    # w2,h2  : [N_COLS]    LN params for in_10
    N_ROWS = in_8.shape[0]   # 300
    K      = in_8.shape[-1]  # 256
    N_COLS = w3.shape[0]     # 256

    out       = torch.empty_like(in_8)
    dtype_str = str(in_8.dtype).split('.')[-1]

    fused_linear_ln_sig_kernel[(N_ROWS,)](
        in_8, in_7, in_6,
        in_9, in_11, in_10,
        w1, h1, w2, h2, w3, h3,
        out,
        K=K,
        N_COLS=N_COLS,
        EPS=1e-5,
        DTYPE_STR=dtype_str,
    )
    return out


def replacement_func():
    return fused_linear_ln_sig