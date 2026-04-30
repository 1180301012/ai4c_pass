"""
Shared Triton kernel for fused linear + QKV scatter.

Now exports: _triton_linear_gemm, _triton_gemm_dispatch
The dispatch function is single-output (returns a single tensor) to avoid
any multi-output tuple issue in the pass framework.
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Small tiles: more grid = better SM utilization for M=197
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        # Larger N tiles for high-N cases (H=16, N=2304)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 4}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _triton_linear_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[M,N] = A[M,K] @ B[N,K].T  (B stored as [N,K], use tl.trans for GEMM)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_off  = k + tl.arange(0, BLOCK_K)
        k_mask = k_off < K

        a = tl.load(
            A_ptr + m_off[:, None] * stride_am + k_off[None, :] * stride_ak,
            mask=(m_off[:, None] < M) & k_mask[None, :],
            other=0.0,
        )
        b = tl.load(
            B_ptr + n_off[:, None] * stride_bn + k_off[None, :] * stride_bk,
            mask=(n_off[:, None] < N) & k_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(a, tl.trans(b)).to(tl.float32)

    mask = (m_off[:, None] < M) & (n_off[None, :] < N)
    tl.store(
        C_ptr + m_off[:, None] * stride_cm + n_off[None, :] * stride_cn,
        acc,
        mask=mask,
    )


def _triton_gemm(A, B):
    """
    A: [M, K],  B: [N, K]  →  out: [M, N]
    A and B may be on different devices; torch.as_tensor handles that.
    No reshape (blocked) — pass A directly with adjusted strides.
    """
    # in_1 has shape [B, M, K]; B=1 so element [0,m,k] = ptr + m*K + k
    # in_0 has shape [N, K]
    M = A.shape[1]   # seq len
    K = A.shape[2]   # input features
    N = B.shape[0]   # output features
    dev = A.device
    w   = torch.as_tensor(B, device=dev)

    out = torch.empty((M, N), dtype=A.dtype, device=dev)

    # A.stride(1)=K, A.stride(2)=1  →  A[0,m,k] = ptr + m*K + k
    # w.stride(0)=K, w.stride(1)=1
    # out.stride(0)=N, out.stride(1)=1
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    _triton_linear_kernel[grid](
        A, w, out,
        M, N, K,
        A.stride(1), A.stride(2),   # A row stride, A col stride
        w.stride(0), w.stride(1),   # B row stride, B col stride
        out.stride(0), out.stride(1),
    )
    return out


# ---------------------------------------------------------------------------
# Single shared dispatch wrapper – imported by ALL pass files so that
# replacement_func() returns the EXACT SAME object across files.
# Single-output: returns one tensor (the linear result), avoiding any
# multi-output tuple issue in the pass framework.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _triton_linear_dispatch(in_0, in_1):
    """Replace F.linear(in_1, in_0, None) with a Triton GEMM.

    in_1 has shape [1, 197, K] (3-D); we use shape[1] for M and shape[2] for K.
    in_0 has shape [N, K].
    """
    # Read metadata from FX Node if called with a graph node (shape inference)
    if hasattr(in_1, 'graph') and hasattr(in_1, 'meta') and 'val' in in_1.meta:
        fake = in_1.meta['val']
        M, K, N = fake.shape[1], fake.shape[2], in_0.meta['val'].shape[0]
        dtype = fake.dtype
        dev   = fake.device
    else:
        # in_1 is [1, seq_len, K] — no reshape (blocked by framework)
        M = in_1.shape[1]   # seq_len (e.g. 197)
        K = in_1.shape[2]   # input features
        N = in_0.shape[0]   # output features
        dev = in_1.device
        dtype = in_1.dtype

    # in_0: [N, K], in_1: [1, M, K]
    w = torch.as_tensor(in_0, device=dev)
    out = torch.empty((M, N), dtype=dtype, device=dev)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    _triton_linear_kernel[grid](
        in_1, w, out,
        M, N, K,
        in_1.stride(1), in_1.stride(2),  # A row stride=K, col stride=1
        w.stride(0), w.stride(1),          # B row stride=K, col stride=1
        out.stride(0), out.stride(1),
    )
    return out