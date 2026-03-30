import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: linear → reshape([-1,9,1]) → softmax(dim=1)
# ---------------------------------------------------------------------------
def pattern(bias, weight, x):
    linear = torch.nn.functional.linear(x, weight, bias)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(bias, weight, x):
    return (bias, weight, x)


# ---------------------------------------------------------------------------
# Fused kernel: linear + grouped-softmax
#
# Grid = (GROUPS_PER_ROW,) = 2 CUDA blocks running IN PARALLEL on 2 SMs.
# Block pid handles column group pid (cols [pid*9:(pid+1)*9]) for ALL M rows.
#
# Tiles (power-of-2, all ≥16 for Ampere tensor-core compatibility):
#   BLOCK_M = 32  (≥ M=19)
#   BLOCK_G = 16  (≥ GROUP_SIZE=9)
#   BLOCK_K = 128 (= K)
#
# tl.dot uses Ampere tensor cores; fp32 accumulator.
# Stores directly in target dtype (no extra conversion kernel).
# num_warps=2, num_stages=1 gives minimum block management overhead.
# ---------------------------------------------------------------------------
@triton.jit
def _fused_linear_softmax(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    GROUP_SIZE,              # runtime: 9
    BLOCK_M: tl.constexpr,  # 32
    BLOCK_G: tl.constexpr,  # 16
    BLOCK_K: tl.constexpr,  # 128
    OUT_DTYPE: tl.constexpr,
):
    pid       = tl.program_id(0)
    col_start = pid * GROUP_SIZE   # 0 or 9

    m_off = tl.arange(0, BLOCK_M)
    g_off = tl.arange(0, BLOCK_G)
    k_off = tl.arange(0, BLOCK_K)

    m_mask = m_off < M
    g_mask = g_off < GROUP_SIZE   # compile-time mask since GROUP_SIZE is constexpr
    k_mask = k_off < K

    # ------------------------------------------------------------------
    # Load x tile: [BLOCK_M, BLOCK_K] = [32, 128]
    # ------------------------------------------------------------------
    x_tile = tl.load(
        x_ptr + m_off[:, None] * stride_xm + k_off[None, :] * stride_xk,
        mask=m_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    # ------------------------------------------------------------------
    # Load w tile for this column group: [BLOCK_G, BLOCK_K] = [16, 128]
    # ------------------------------------------------------------------
    w_tile = tl.load(
        w_ptr + (col_start + g_off)[:, None] * stride_wn
              + k_off[None, :] * stride_wk,
        mask=g_mask[:, None] & k_mask[None, :],
        other=0.0,
    )

    # ------------------------------------------------------------------
    # GEMM [32,128]×[128,16] → [32,16] fp32  (Ampere tensor cores)
    # ------------------------------------------------------------------
    acc = tl.zeros([BLOCK_M, BLOCK_G], dtype=tl.float32)
    acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=True)

    # ------------------------------------------------------------------
    # Add bias
    # ------------------------------------------------------------------
    b_vals = tl.load(b_ptr + col_start + g_off,
                     mask=g_mask, other=0.0).to(tl.float32)
    acc = acc + b_vals[None, :]

    # ------------------------------------------------------------------
    # Numerically-stable row-wise softmax over GROUP_SIZE columns
    # ------------------------------------------------------------------
    NEG_BIG = -1e9
    acc_max  = tl.max(tl.where(g_mask[None, :], acc, NEG_BIG), axis=1)
    acc_exp  = tl.exp(acc - acc_max[:, None])
    acc_exp  = tl.where(g_mask[None, :], acc_exp, 0.0)
    acc_sum  = tl.sum(acc_exp, axis=1)
    acc_out  = acc_exp / acc_sum[:, None]

    # ------------------------------------------------------------------
    # Store: out_flat[m*N + col_start + g]  (N is compile-time → fast multiply)
    # ------------------------------------------------------------------
    out_off  = m_off[:, None] * N + (col_start + g_off[None, :])
    out_mask = m_mask[:, None] & g_mask[None, :]
    tl.store(out_ptr + out_off, acc_out.to(OUT_DTYPE), mask=out_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_linear_reshape_softmax(bias, weight, x):
    orig_dtype = x.dtype

    x_flat = x.view(-1, x.shape[-1])   # [M, K]
    M = x_flat.shape[0]                # 19
    K = x_flat.shape[1]                # 128
    N = weight.shape[0]                # 18

    GROUP_SIZE     = 9
    GROUPS_PER_ROW = N // GROUP_SIZE   # 2

    BLOCK_M = triton.next_power_of_2(M)   # 32
    BLOCK_G = 16
    BLOCK_K = triton.next_power_of_2(K)   # 128

    OUT_DTYPE = tl.bfloat16 if orig_dtype == torch.bfloat16 else tl.float16

    out_flat = torch.empty(M * N, dtype=orig_dtype, device=x.device)

    _fused_linear_softmax[(GROUPS_PER_ROW,)](   # 2 parallel blocks
        x_flat, weight, bias, out_flat,
        M, K, N,
        x_flat.stride(0), x_flat.stride(1),
        weight.stride(0), weight.stride(1),
        GROUP_SIZE,
        BLOCK_M=BLOCK_M,
        BLOCK_G=BLOCK_G,
        BLOCK_K=BLOCK_K,
        OUT_DTYPE=OUT_DTYPE,
        num_warps=2,
        num_stages=1,
    )

    return out_flat.view(-1, GROUP_SIZE, 1)


def replacement_func():
    return fused_linear_reshape_softmax