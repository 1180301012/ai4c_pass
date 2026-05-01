"""
Shared linear implementation used by both pattern-replacement passes.
Both FuseLinearSplitUnsqueeze and FuseLinearSplit3D import
dispatch_fused_linear_split from this file so they return the SAME Python
function object, satisfying output_pass_replacement_func_limit == 1.

Design note
-----------
Pattern: torch.nn.functional.linear(x, w, b) → single full-output tensor.
  - This is a SINGLE-OUTPUT pattern (1 returning node), which is required for
    the custom_replacement._replace_pattern assertion to pass.
  - Downstream slice / view / unsqueeze ops remain in the graph as zero-copy
    metadata operations.
Replacement: Triton GEMM kernel that writes the full [M, N] output.
  Works for both 2-D and ND inputs by flattening leading dimensions.
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Triton GEMM kernel: out[m, n] = sum_k  x[m,k] * w[n,k]  + b[n]
#   w is loaded transposed to avoid tl.trans.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # Smaller BLOCK_M → more blocks (better SM utilization for M=300)
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=4, num_warps=8),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, K, N,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    IS_FP16: tl.constexpr,
    IS_BF16: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # x: [BLOCK_M, BLOCK_K] — coalesced (K dim is innermost)
        x = tl.load(
            x_ptr + m_offs[:, None] * stride_xm + k_offs[None, :] * stride_xk,
            mask=(m_offs[:, None] < M) & (k_offs[None, :] < K),
            other=0.0,
        )

        # w: [BLOCK_N, BLOCK_K] — coalesced (K dim is innermost for each N row)
        # Then tl.trans(w) → [BLOCK_K, BLOCK_N] for the matmul
        w = tl.load(
            w_ptr + n_offs[:, None] * stride_wn + k_offs[None, :] * stride_wk,
            mask=(n_offs[:, None] < N) & (k_offs[None, :] < K),
            other=0.0,
        )

        # [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N] = [BLOCK_M, BLOCK_N]
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)

    # Add bias
    b = tl.load(b_ptr + n_offs, mask=n_offs < N, other=0.0)
    acc += b[None, :].to(tl.float32)

    m_mask = m_offs[:, None] < M
    n_mask = n_offs[None, :] < N

    if IS_FP16:
        out = acc.to(tl.float16)
    elif IS_BF16:
        out = acc.to(tl.bfloat16)
    else:
        out = acc

    tl.store(
        out_ptr + m_offs[:, None] * stride_om + n_offs[None, :] * stride_on,
        out,
        mask=m_mask & n_mask,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Shared dispatch wrapper  (MUST be @torch.fx.wrap)
# Replaces F.linear for any ND input by flattening leading dims.
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def dispatch_fused_linear_split(x, w, b, route):
    """Triton-accelerated linear: out = x @ w.T + b  (any ND input)."""
    *leading, K = x.shape
    N = w.shape[0]
    M = 1
    for d in leading:
        M = M * d

    out = torch.empty((*leading, N), dtype=x.dtype, device=x.device)

    is_fp16 = (x.dtype == torch.float16)
    is_bf16 = (x.dtype == torch.bfloat16)
    N_val = N

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_val, meta['BLOCK_N']),
    )
    _linear_kernel[grid](
        x, w, b, out,
        M, K, N,
        x.stride(-2), x.stride(-1),
        w.stride(0), w.stride(1),
        out.stride(-2), out.stride(-1),
        IS_FP16=is_fp16,
        IS_BF16=is_bf16,
    )
    return out