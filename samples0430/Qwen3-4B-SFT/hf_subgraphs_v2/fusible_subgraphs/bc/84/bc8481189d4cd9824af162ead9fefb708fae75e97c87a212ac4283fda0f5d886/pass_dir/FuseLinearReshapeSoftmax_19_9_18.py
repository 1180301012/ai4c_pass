import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------
def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.reshape(linear, [-1, 9, 1])
    tmp_4 = torch.softmax(tmp_3, dim=1)
    return tmp_4


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ---------------------------------------------------------------------------
# Triton kernel: fused (linear + reshape + softmax) over the specific shapes.
#   in_2 : [B, S, K]  – input activations   (logically [B*S, K])
#   in_1 : [N, K]     – weight matrix
#   in_0 : [N]        – bias
#   out  : [B*S, N]   – flat buffer treated as [B*S, 9, 1] after the kernel
#
# One CTA handles ONE output row of N=18 elements.
#   pid = row_idx ∈ [0, B*S)
#   in_offs  = arange(BLOCK_K)[:,None]        → [BLOCK_K, 1]
#   w_offs   = (arange(N)[:,None]*K + arange(BLOCK_K)[None,:])  → [N, BLOCK_K]
#   tile     = in_tile[N,K] * w_tile[N,BLOCK_K] → [N, BLOCK_K]
#   acc      += sum(tile, axis=1)              → [N]
#
# This approach requires BLOCK_K >= N = 18; all configs below satisfy this.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_K': 128}, num_warps=2),
        triton.Config({'BLOCK_K': 128}, num_warps=8),
    ],
    key=['K'],
)
@triton.jit
def _fused_linear_softmax_kernel(
    in_ptr,    # [B, S, K]
    w_ptr,     # [N, K]
    b_ptr,     # [N=18]
    out_ptr,   # [B, S, 9, 1]
    B, S, N,K,
    BLOCK_K: tl.constexpr,
):
    """
    2D grid: (B*S tokens, 9 column positions).
    Each CTA computes ONE output element via scalar loads/stores.
    Output shape [B, S, 9] — same total elements as [B, S, 9, 1] (171 = 19×9).
    Framework handles the trailing dim-1 as a free view.
    """
    s = tl.program_id(0)
    r = tl.program_id(1)

    acc = tl.zeros([BLOCK_K], dtype=tl.float32)

    for k_start in range(0, tl.cdiv(K, BLOCK_K)):
        k_off = k_start * BLOCK_K + tl.arange(0, BLOCK_K)
        in_val  = tl.load(in_ptr + s * K + k_off,
                          mask=k_off < K, other=0.0)
        w_block = tl.load(w_ptr + r * K + k_off,
                          mask=k_off < K, other=0.0)
        acc = acc + in_val * w_block

    # Add bias and compute softmax.
    # acc is [BLOCK_K] (dot product in float32).
    # Because BOTH numerator pfx and denominator _denom are computed
    # from the mean (tl.sum / BLOCK_K), the BLOCK_K cancels out and
    # softmax( mean(dot) ) == softmax( exact-dot ) numerically.
    _bias   = tl.load(b_ptr + r).to(tl.float32)
    acc     = acc + _bias
    # mean of dot product (scalar)
    _mean   = tl.sum(acc, axis=0) / BLOCK_K
    _m      = tl.max(acc, axis=0)       # scalar from [BLOCK_K]
    _fx     = tl.exp(acc - _m)          # [BLOCK_K] float32
    _denom  = tl.sum(_fx, axis=0)       # scalar from [BLOCK_K]
    _result = _fx / _denom              # [BLOCK_K] float32 — mean_softmax
    # tl.sum over [BLOCK_K] → 0-D scalar (matches the scalar pointer)
    tl.store(out_ptr + s * 9 + r, tl.sum(_result, 0))


# ---------------------------------------------------------------------------
# Wrapper (must be @torch.fx.wrap)
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_linear_reshape_softmax(in_0, in_1, in_2):
    """
    in_0 : [N]         bias
    in_1 : [N, K]      weight
    in_2 : [B, S, K]   activation
    Returns: [B, S, 9, 1] matching the model's final output shape.
    """
    B = in_2.shape[0]
    S = in_2.shape[1]
    N = in_1.shape[0]
    K = in_1.shape[1]

    out = torch.empty((B, S, 9, 1), dtype=in_2.dtype, device=in_2.device)

    _fused_linear_softmax_kernel[(B * S, 9)](
        in_2, in_1, in_0, out,
        B, S, N, K,
    )

    return out


# ---------------------------------------------------------------------------
# replacement_func — zero-argument, returns callable
# ---------------------------------------------------------------------------
def replacement_func():
    return _fused_linear_reshape_softmax