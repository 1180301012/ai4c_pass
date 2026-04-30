"""
Shared Triton kernels for matmul-reshape and transpose passes.
"""
import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Matmul-reshape kernel
# out[b, m] = sum_k(A[b, m, k] * C[b, k, 0])
# A=[B, M, K], C=[B, K, 1], out=[B, M]
# BLOCK_N (= M) passed explicitly as constexpr, autotune only varies BLOCK_M/BLOCK_K.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16}, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'K'],
)
@triton.jit
def _matmul_reshape_kernel(
    A_ptr, C_ptr, out_ptr,
    B, M, K, N,
    stride_Ab, stride_Am, stride_Ak,
    stride_Cb, stride_Ck,
    stride_ob, stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)                    # [BLOCK_N]

    mask_m = offs_m < M
    mask_n = offs_n < N

    # 1-D accumulator: acc[m] = sum_k A[m,k]*C[k]
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # A tile: [BLOCK_M, BLOCK_K]
        A = tl.load(
            A_ptr + pid_b * stride_Ab
                   + offs_m[:, None] * stride_Am
                   + k_offs[None, :] * stride_Ak,
            mask=mask_m[:, None] & k_mask[None, :],
            other=0.0,
        )

        # C[k_offs, 0]: [BLOCK_K]
        C = tl.load(
            C_ptr + pid_b * stride_Cb + k_offs * stride_Ck,
            mask=k_mask, other=0.0,
        )

        # acc[m] += sum_k A[m, k] * C[k]
        acc += tl.sum(A * C[None, :], axis=1)

    # Store: acc[m] -> out[b, m, n] for all n in [0, N)
    out_ptrs = out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :]
    ones_n = tl.full((1, BLOCK_N), 1.0, dtype=tl.float32)
    store_val = (acc[:, None] * ones_n).to(A_ptr.dtype.element_ty)
    tl.store(out_ptrs, store_val, mask=mask_m[:, None] & mask_n[None, :])


# ---------------------------------------------------------------------------
# Transpose kernel  (last-two dims of a 4-D tensor)
# in_2: [..., M, N]  →  out: [..., N, M]
# Uses tl.trans with coalesced loads (n is fast axis) and coalesced stores.
# ---------------------------------------------------------------------------

@triton.jit
def _transpose_last2_kernel(
    in_ptr, out_ptr,
    M, N,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # [BLOCK_N]

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load [BLOCK_M, BLOCK_N] (coalesced: n is fast axis, stride 1)
    in_ptrs = in_ptr + offs_m[:, None] * N + offs_n[None, :]
    in_tile = tl.load(in_ptrs, mask=mask_m[:, None] & mask_n[None, :], other=0.0)
    # in_tile: [BLOCK_M, BLOCK_N]

    # out_mask for transposed [BLOCK_N, BLOCK_M] store
    out_mask = mask_n[:, None] & mask_m[None, :]

    # Store transposed: out[n_i, m_i] = in[m_i, n_i]
    # out_ptrs[j, i] = offs_n[j]*M + offs_m[i]  (coalesced: m is fast axis, stride 1)
    out_ptrs = out_ptr + offs_n[:, None] * M + offs_m[None, :]
    tl.store(out_ptrs, tl.trans(in_tile), mask=out_mask)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper (imported and reused by each pass file)
# route encodes which sub-operation to run.
# ---------------------------------------------------------------------------

@torch.fx.wrap
def shared_dispatch(a, b, c, route):
    if route == "matmul_reshape":
        # a = in_1  [B, M, K]
        # b = in_0  [B, K, 1]
        B = a.shape[0]
        M = a.shape[1]
        K = a.shape[2]
        N = M   # reshape output is [-1, M]
        out = torch.empty((B, M), dtype=a.dtype, device=a.device)
        grid = lambda meta: (B, triton.cdiv(M, meta['BLOCK_M']))
        _matmul_reshape_kernel[grid](
            a, b, out,
            B, M, K, N,
            a.stride(0), a.stride(1), a.stride(2),
            b.stride(0), b.stride(1),
            out.stride(0), out.stride(1),
            BLOCK_N=N,   # passed explicitly; also in autotune config
        )
        return out.reshape(-1, M)

    elif route == "transpose":
        # c = in_2  [..., M, N]
        s = c.shape
        ndim = len(s)
        M_val = s[ndim - 2]
        N_val = s[ndim - 1]
        out = torch.empty(ndim, dtype=c.dtype, device=c.device)
        out = torch.empty((*s[:-2], N_val, M_val), dtype=c.dtype, device=c.device)
        BLOCK_M = 8
        BLOCK_N = 8
        num_pid_m = triton.cdiv(M_val, BLOCK_M)
        num_pid_n = triton.cdiv(N_val, BLOCK_N)
        grid = (num_pid_m * num_pid_n,)
        _transpose_last2_kernel[grid](
            c, out, M_val, N_val,
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )
        return out

    else:
        # Should never reach here
        return c