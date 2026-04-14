"""
Shared Triton batched-matmul kernel used by both pass files.
Computes C = in_1 @ in_0.

All inputs are 4-D: [B, H, M, K] (in_1) and [B, H, K, N] (in_0).
The batch dimensions B and H are folded into a single flat batch
dimension B*H by using stride(1) as the per-element batch stride:

  For a 4-D contiguous (or batch-contiguous) tensor of shape [B, H, *, *]:
      element[b_flat] starts at  ptr + b_flat * stride(1)
  because stride(0) == H * stride(1), so
      b * stride(0) + h * stride(1) == (b*H + h) * stride(1) == b_flat * stride(1).

This also works for tensors that are transposed only in the last two dims
(like the attention-weight tensor `transpose_1`).

Output shape: [B*H, M, N]  – the model's downstream .view() reshapes it further.
Only torch.empty() is used inside the replacement wrapper; no other aten ops.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Triton kernel
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # Balanced tiles – YOLO style (M=64, K=400, N=400)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Tall-and-thin – GCNet / ViPNAS style (large M, N=1..48)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        # Small fallback
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 16,  'BLOCK_K': 16}, num_stages=3, num_warps=2),
        # Wide tiles for high-batch scenarios
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _batched_matmul_kernel(
    A, B, C,
    M, K, N,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]"""
    b_id  = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_base = A + b_id * stride_ab
    B_base = B + b_id * stride_bb
    C_base = C + b_id * stride_cb

    A_ptrs = A_base + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_base + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_rem = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_rem)
        b_mask = (offs_k[:, None] < k_rem) & (offs_n[None, :] < N)

        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_base + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(C.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrapper  (decorated with @torch.fx.wrap so FX treats it as a leaf)
# Only torch.empty() is used; all other ops are pure Python on Python objects
# (.shape, .stride(), .dtype, .device return Python scalars/tuples, no aten).
# ---------------------------------------------------------------------------

@torch.fx.wrap
def triton_batched_matmul(in_0, in_1):
    """
    Drop-in replacement for  result = in_1 @ in_0.

    in_1 : [B, H, M, K]   (4-D)
    in_0 : [B, H, K, N]   (4-D)
    out  : [B*H, M, N]    – downstream .view() in the model reshapes further.
    """
    # -- shape & stride info (Python scalars, no aten dispatch) --
    M = in_1.shape[-2]
    K = in_1.shape[-1]
    N = in_0.shape[-1]

    # Flat batch size: product of all leading dims  (pure Python arithmetic)
    batch = 1
    for d in in_1.shape[:-2]:
        batch *= d

    # Flat-batch stride for 4-D inputs:
    #   stride(1) is the H-dimension stride.
    #   For b_flat = b*H + h:  offset = b_flat * stride(1)  (see module docstring).
    stride_ab = in_1.stride(1)
    stride_bb = in_0.stride(1)

    # Allocate output (only whitelisted torch.* call)
    c = torch.empty((batch, M, N), dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        batch,
    )

    _batched_matmul_kernel[grid](
        in_1, in_0, c,
        M, K, N,
        stride_ab,      in_1.stride(-2), in_1.stride(-1),
        stride_bb,      in_0.stride(-2), in_0.stride(-1),
        c.stride(0),    c.stride(1),     c.stride(2),
    )

    return c