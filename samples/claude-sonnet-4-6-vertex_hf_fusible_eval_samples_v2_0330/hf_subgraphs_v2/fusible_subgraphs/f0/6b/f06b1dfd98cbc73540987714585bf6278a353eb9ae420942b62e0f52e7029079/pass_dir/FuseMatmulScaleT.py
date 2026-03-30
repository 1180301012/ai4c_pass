import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: matmul(in_2, in_1) * in_0
#   in_0 : scalar  (logit_scale)
#   in_1 : [K, 1]
#   in_2 : [M, K]
#   output tmp_1 : [M, 1]
#
# .t() stays in the graph (free view, not part of pattern).
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    matmul = torch.matmul(in_2, in_1)
    tmp_1 = matmul * in_0
    return tmp_1


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel – single program (grid=1), 2D block.
#
# Grid  : (1,)
# Block : [BLOCK_M, BLOCK_K]  – all M rows × all K cols at once
#
# Benefits vs. per-row approach:
#   • in_1 is loaded only ONCE (shared across all rows)
#   • Single CTA → minimal GPU scheduling overhead
#   • Direct output to input dtype (no extra cast kernel)
# ---------------------------------------------------------------------------
@triton.jit
def _fused_gemv_scale_2d(
    in2_ptr,              # [M, K]  row-major
    in1_ptr,              # [K, 1]  (element [k,0] = ptr+k, stride=1)
    in0_ptr,              # scalar
    out_ptr,              # [M, 1]  (element [m,0] = ptr+m, contiguous)
    M, K,
    BLOCK_M: tl.constexpr,    # next_power_of_2(M)   e.g. 2
    BLOCK_K: tl.constexpr,    # next_power_of_2(K)   e.g. 1024
    IS_BF16: tl.constexpr,
    IS_FP16: tl.constexpr,
):
    # ---- load scale (scalar) ----
    scale = tl.load(in0_ptr).to(tl.float32)

    m_offs = tl.arange(0, BLOCK_M)      # [0..M-1,...]
    k_offs = tl.arange(0, BLOCK_K)      # [0..K-1,...]
    mask_m = m_offs < M
    mask_k = k_offs < K

    # ---- load in2 as 2D tile [BLOCK_M, BLOCK_K] ----
    a = tl.load(
        in2_ptr + m_offs[:, None] * K + k_offs[None, :],
        mask=mask_m[:, None] & mask_k[None, :],
        other=0.0,
    ).to(tl.float32)                    # [BLOCK_M, BLOCK_K]

    # ---- load in1[:,0] as 1D tile [BLOCK_K] (shared by all rows) ----
    b = tl.load(
        in1_ptr + k_offs,
        mask=mask_k,
        other=0.0,
    ).to(tl.float32)                    # [BLOCK_K]

    # ---- row-wise dot products → [BLOCK_M] ----
    # tl.sum([M, K] * [1, K], axis=1) → [M]  (each row dot col)
    results = tl.sum(a * b[None, :], axis=1) * scale   # [BLOCK_M]

    # ---- in-register dtype cast (no extra kernel) ----
    if IS_BF16:
        results_out = results.to(tl.bfloat16)
    elif IS_FP16:
        results_out = results.to(tl.float16)
    else:
        results_out = results   # already float32

    # ---- store M values; out[m,0] = ptr+m for [M,1] contiguous layout ----
    tl.store(out_ptr + m_offs, results_out, mask=mask_m)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def fused_matmul_scale(in_0, in_1, in_2):
    """
    Single Triton kernel fusing GEMV + scalar multiply.
    Output: [M, 1] in the same dtype as the inputs.
    """
    M = in_2.shape[0]    # 2
    K = in_2.shape[1]    # 1024

    out = torch.empty(M, 1, dtype=in_2.dtype, device=in_2.device)

    BLOCK_M = triton.next_power_of_2(M)   # 2
    BLOCK_K = triton.next_power_of_2(K)   # 1024

    _fused_gemv_scale_2d[(1,)](
        in_2, in_1, in_0, out,
        M, K,
        BLOCK_M=BLOCK_M,
        BLOCK_K=BLOCK_K,
        IS_BF16=(in_2.dtype == torch.bfloat16),
        IS_FP16=(in_2.dtype == torch.float16),
    )

    return out


# ---------------------------------------------------------------------------
# Required by the pass framework
# ---------------------------------------------------------------------------
def replacement_func():
    return fused_matmul_scale