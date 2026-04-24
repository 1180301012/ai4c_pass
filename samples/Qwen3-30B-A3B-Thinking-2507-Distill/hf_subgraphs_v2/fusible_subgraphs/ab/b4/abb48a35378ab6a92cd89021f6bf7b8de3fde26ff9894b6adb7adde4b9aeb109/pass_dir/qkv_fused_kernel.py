import torch
import triton
import triton.language as tl


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: GEMM (X @ W^T) with direct scatter to [3, 1, H, S, D] layout.
# Fuses: linear(in_1,in_0,None).reshape(1,S,3,H,D).permute(2,0,3,1,4)
# ──────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['S', 'N', 'K'],
)
@triton.jit
def _qkv_linear_permute_kernel(
    x_ptr, w_ptr, out_ptr,
    S, N, K, H, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # seq offsets [BLOCK_M]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # out-channel offsets [BLOCK_N]

    # ── tiled GEMM: acc[m,n] = sum_k X[m,k] * W[n,k] ──────────────────────
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # X tile: [BLOCK_M, BLOCK_K]  (coalesced along k)
        x = tl.load(
            x_ptr + m_offs[:, None] * K + k_offs[None, :],
            mask=(m_offs[:, None] < S) & (k_offs[None, :] < K),
            other=0.0,
        )
        # W tile: [BLOCK_N, BLOCK_K]  (coalesced along k)
        w = tl.load(
            w_ptr + n_offs[:, None] * K + k_offs[None, :],
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )
        # x @ w^T: [BLOCK_M,BLOCK_K] @ [BLOCK_K,BLOCK_N] = [BLOCK_M,BLOCK_N]
        acc += tl.dot(x, tl.trans(w))

    # ── cast accumulator to the input/output element type ──────────────────
    # Use x_ptr.dtype.element_ty (static pointer attribute, always in scope)
    acc_out = acc.to(x_ptr.dtype.element_ty)

    # ── Decompose n into (qkv-type, head-index, head-dim-index) ────────────
    qkv   = n_offs // (D * H)   # 0=Q, 1=K^T, 2=V
    rem   = n_offs % (D * H)
    h_idx = rem  // D
    d_idx = rem  % D

    # Output tensor shape: [3, 1, H, S, D] (contiguous), strides (H*S*D, H*S*D, S*D, D, 1)
    # Element (qkv, 0, h, s, d) → flat offset = qkv*H*S*D + h*S*D + s*D + d
    out_off = qkv[None, :] * (H * S * D) + h_idx[None, :] * (S * D) + m_offs[:, None] * D + d_idx[None, :]
    m_ok    = (m_offs < S)                        # [BLOCK_M]
    n_ok    = (n_offs < N)                        # [BLOCK_N]

    # ── Store Q (qkv==0) ────────────────────────────────────────────────────
    q_mask = ((n_offs == 0)[None, :] & m_ok[:, None] & n_ok[None, :])
    tl.store(out_ptr + out_off, acc_out, mask=q_mask)

    # ── Store K^T (qkv==1): layout (H,D,S) → offset += h*S*D + d*S + s ───
    kt_off  = out_off + h_idx[None, :] * (S * D) + d_idx[None, :] * S + m_offs[:, None]
    kt_mask = ((n_offs == 1)[None, :] & m_ok[:, None] & n_ok[None, :])
    tl.store(out_ptr + kt_off, acc_out, mask=kt_mask)

    # ── Store V (qkv==2) — same layout as Q ─────────────────────────────────
    v_mask = ((n_offs == 2)[None, :] & m_ok[:, None] & n_ok[None, :])
    tl.store(out_ptr + out_off, acc_out, mask=v_mask)


# ──────────────────────────────────────────────────────────────────────────────
# Single-output replacement wrapper.
# All pass files import and return this SAME function so replacement_func_limit
# is satisfied (one unique body across all passes).
# ──────────────────────────────────────────────────────────────────────────────
def _kv_linear_permute(in_0, in_1):
    """
    Replaces:  linear(in_1, in_0, None).reshape(1, 197, 3, H, 48).permute(2,0,3,1,4)
    with a single fused Triton GEMM writing to [3, 1, H, S, D] layout.

    in_0 : weight [N, K]   (N = 3 * H * 48)
    in_1 : input  [1, S, K]
    Returns contiguous [3, 1, H, S, D].
    """
    D = 48
    S = 197
    H = in_0.shape[0] // (3 * D)
    K = in_1.shape[-1]
    N = 3 * H * D
    M = S

    # Allocate [3, 1, H, S, D] — same contiguous layout as permute(2,0,3,1,4)
    out = torch.empty((3, 1, H, S, D), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']),
                         triton.cdiv(N, meta['BLOCK_N']))

    _qkv_linear_permute_kernel[grid](
        in_1, in_0, out,
        M, N, K, H, D,
    )
    return out