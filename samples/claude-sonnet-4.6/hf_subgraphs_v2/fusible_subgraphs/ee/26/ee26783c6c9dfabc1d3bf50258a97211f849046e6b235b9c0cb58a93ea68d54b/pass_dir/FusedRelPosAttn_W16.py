import torch
import triton
import triton.language as tl


# ── Fused (x+bias).softmax @ y transposed ────────────────────────────────────
#  Single-pass: loads all N attention-score columns at once, computes full
#  row-wise softmax, then does one tl.dot([BQ, N] × [N, D]) = [BQ, D].
#  Grid: (B * N//BQ,)  – 1D, no D-tiling to maximise D-reuse.

@triton.autotune(
    configs=[
        triton.Config({'BQ': 16}, num_warps=4),
        triton.Config({'BQ': 32}, num_warps=4),
        triton.Config({'BQ': 32}, num_warps=8),
    ],
    key=['N', 'D'],
)
@triton.jit
def _add_sm_gemm_t(
    x_ptr, bias_ptr, y_ptr, out_ptr,
    B: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
    BQ: tl.constexpr,
):
    bq_pid  = tl.program_id(0)
    bid     = bq_pid // (N // BQ)
    q_block = bq_pid %  (N // BQ)
    q_start = q_block * BQ

    q_range = q_start + tl.arange(0, BQ)
    k_range = tl.arange(0, N)
    d_range = tl.arange(0, D)

    # ── Load (x + bias)[bid, q_range, :] → [BQ, N] ────────────────────────
    base   = bid * (N * N) + q_range[:, None] * N + k_range[None, :]
    scores = (tl.load(x_ptr    + base).to(tl.float32) +
              tl.load(bias_ptr + base).to(tl.float32))  # [BQ, N]

    # ── Row-wise softmax → cast to native dtype ────────────────────────────
    row_max = tl.max(scores, axis=1)[:, None]
    exp_s   = tl.exp(scores - row_max)
    prob    = (exp_s / tl.sum(exp_s, axis=1)[:, None]).to(y_ptr.dtype.element_ty)  # [BQ, N]

    # ── Load value matrix y[bid, :, :] → [N, D] ───────────────────────────
    y_ptrs  = y_ptr + bid * (N * D) + k_range[:, None] * D + d_range[None, :]
    y_full  = tl.load(y_ptrs)   # [N, D]

    # ── Single tl.dot: [BQ, N] × [N, D] → [BQ, D] ────────────────────────
    out = tl.dot(prob, y_full, allow_tf32=False)   # [BQ, D] fp32

    # ── Store transposed: out[bid, d, q_range] ─────────────────────────────
    out_ptrs = out_ptr + bid * (D * N) + d_range[:, None] * N + q_range[None, :]
    tl.store(out_ptrs, tl.trans(out).to(x_ptr.dtype.element_ty))


@torch.fx.wrap
def fused_add_softmax_matmul_t(x, bias, y):
    B = x.shape[0];  N = x.shape[1];  D = y.shape[2]
    out = torch.empty((B, D, N), dtype=x.dtype, device=x.device)
    def grid(meta):
        return (B * (N // meta['BQ']),)
    _add_sm_gemm_t[grid](x, bias, y, out, B=B, N=N, D=D)
    return out


def pattern(in_0, tmp_11, in_4):
    tmp_12   = in_0 + tmp_11
    tmp_13   = tmp_12.softmax(dim=-1)
    matmul_1 = tmp_13 @ in_4
    tmp_15   = matmul_1.transpose(-1, -2)
    return tmp_15


def replacement_args(in_0, tmp_11, in_4):
    return (in_0, tmp_11, in_4)


def replacement_func():
    return fused_add_softmax_matmul_t