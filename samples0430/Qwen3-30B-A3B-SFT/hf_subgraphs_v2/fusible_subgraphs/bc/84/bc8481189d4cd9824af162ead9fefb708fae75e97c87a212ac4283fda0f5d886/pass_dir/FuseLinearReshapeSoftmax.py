import torch
import triton
import triton.language as tl


@triton.jit
def fused_linear_reshape_softmax_kernel(
    x_ptr,        # input: [B*S, K]  (flat: [19, 128])
    w_ptr,        # weight: [2*N, K] = [18, 128]
    b_ptr,        # bias:   [2*N]    = [18]
    out_ptr,      # output: [B*S, 2*N, 1]  (3-D; strides [2*N, 1, 1])
    S: tl.constexpr,        # sequence length = 19
    N: tl.constexpr,        # channels per split = 9
    K: tl.constexpr,        # inner dim = 128
    TOTAL_SEQ: tl.constexpr, # B*S = 19
    BLOCK_M: tl.constexpr,  # 16  (rows per program)
    BLOCK_N: tl.constexpr,  # 32  (≥ 2*N = 18)
    BLOCK_K: tl.constexpr,  # 16  (K-chunk for tl.dot)
):
    """
    One program handles BLOCK_M=16 sequences.
    Uses tl.dot(x_tile [M,K], w_T_tile [K,N]) to compute all 2*N=18 logits
    at once per K-chunk, accumulating into k_block [M, N].
    Then applies masked row-softmax and stores two groups of N=9 to the
    3-D output [B*S, N, 1] using Python slice notation (supported in Triton 2.x).
    """
    pid       = tl.program_id(0)
    seq_start = pid * BLOCK_M
    m_range   = seq_start + tl.arange(0, BLOCK_M)   # [M=16]
    m_mask    = m_range < TOTAL_SEQ                   # valid rows

    k_block = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_idx in range(K // BLOCK_K):
        k_start = k_idx * BLOCK_K
        k_range = k_start + tl.arange(0, BLOCK_K)   # [K=16]

        # x_tile [M, K]: rows of x at seq_start..seq_start+M, cols k_start..k_start+K
        x_tile = tl.load(
            x_ptr + m_range[:, None] * K + k_range[None, :],
            mask=m_mask[:, None], other=0.0,
        ).to(tl.float32)

        # w_T_tile [K, N]: load w^T so that w_T_tile[k, n] = w[n, k]
        n_range = tl.arange(0, BLOCK_N)
        w_T_tile = tl.load(
            w_ptr + n_range[None, :] * K + k_range[:, None],
            mask=(n_range[None, :] < 2 * N), other=0.0,
        ).to(tl.float32)

        # [M, K] @ [K, N] → [M, N]
        k_block = k_block + tl.dot(x_tile, w_T_tile)

    # Add bias (broadcasted)
    n_range = tl.arange(0, BLOCK_N)
    b     = tl.load(b_ptr + n_range, mask=n_range < 2 * N, other=0.0).to(tl.float32)
    k_block = k_block + b[None, :]

    # Mask out padding cols (2*N=18 .. BLOCK_N-1=31) with -inf for correct softmax
    k_block = tl.where(n_range[None, :] < 2 * N, k_block, float('-inf'))

    # Row-wise softmax over all BLOCK_N cols
    m_all = tl.max(k_block, axis=1)              # [M]
    e_all = tl.exp(k_block - m_all[:, None])    # [M, N]
    r_all = e_all / tl.sum(e_all, axis=1)[:, None]  # [M, N]

    # nc_range is [BLOCK_N=32]; use it with masks instead of Python slice notation
    nc_range = tl.arange(0, BLOCK_N)             # [32]

    # Group 0: first N=9 cols → out[m, 0:N, 0]
    n0_mask = nc_range < N                        # first 9 valid
    tl.store(
        out_ptr + m_range[:, None] * (2 * N) + nc_range[None, :],
        r_all.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n0_mask[None, :],
    )

    # Group 1: next N=9 cols → out[m+S, 0:N, 0]
    n1_mask = (nc_range < 2 * N) & (~n0_mask)    # cols N..2N-1 valid
    tl.store(
        out_ptr + (m_range[:, None] + S) * (2 * N) + nc_range[None, :],
        r_all.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n1_mask[None, :],
    )


@torch.fx.wrap
def fused_linear_reshape_softmax(in_0, in_1, in_2):
    """
    Fused replacement for:
        linear = F.linear(in_2, in_1, in_0)    # [1,19,128] x [18,128]^T + [18]
        tmp_3  = reshape(linear, [-1, 9, 1])   # [19, 9, 1]
        tmp_4  = softmax(tmp_3, dim=1)          # [19, 9, 1]
        return tmp_4
    """
    B, S, K = in_2.shape          # B=1, S=19, K=128
    C       = in_1.shape[0]       # 18
    N       = C // 2              # 9
    BLOCK_M = 16
    BLOCK_N = 32
    BLOCK_K = 16

    # Allocate 3-D output [B*S, N, 1] directly — no reshape needed
    out = torch.empty((B * S, N, 1), dtype=in_2.dtype, device=in_2.device)

    grid = ((B * S + BLOCK_M - 1) // BLOCK_M,)   # (2,)

    fused_linear_reshape_softmax_kernel[grid](
        in_2, in_1, in_0, out,
        TOTAL_SEQ=B * S,
        S=S, N=N, K=K,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    return (out,)


# ---------------------------------------------------------------------------
# Pattern / replacement interface required by the AI4C framework
# ---------------------------------------------------------------------------

def pattern(in_0, in_1, in_2):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3  = torch.reshape(linear, [-1, 9, 1])
    tmp_4  = torch.softmax(tmp_3, dim=1)
    return (tmp_4,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


def replacement_func():
    return fused_linear_reshape_softmax