"""
Shared Triton kernels + single dispatch wrapper for ALL passes.

Two independent operations are optimised here:

  1. mean(dim=-2, keepdim=True)   on  [B, N, D]  →  [B, 1, D]
  2. conv2d 1x1 + view            on  [B, Cin, H, W]  →  [B, Cout, H*W]

All pass files import and return `dispatch_wrapper` from replacement_func()
so that only ONE unique replacement function is seen by the framework.
"""

import torch
import triton
import triton.language as tl


# ═══════════════════════════════════════════════════════════════════════════════
#  1.  Mean-reduction kernel   [B, N, D]  →  [B, D]  (caller adds the dim=1)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64,   'BLOCK_D': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128,  'BLOCK_D': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_D': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512,  'BLOCK_D': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_D': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,   'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128,  'BLOCK_D': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512,  'BLOCK_D': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 64,   'BLOCK_D': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 128,  'BLOCK_D': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256,  'BLOCK_D': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 512,  'BLOCK_D': 256}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_D': 256}, num_warps=8, num_stages=3),
    ],
    key=['B', 'N', 'D'],
)
@triton.jit
def _mean_neg2_keepdim_kernel(
    x_ptr, out_ptr,
    B, N, D,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid        = tl.program_id(0)
    num_dblks  = tl.cdiv(D, BLOCK_D)
    b_idx      = pid // num_dblks
    d_blk      = pid %  num_dblks

    if b_idx >= B:
        return

    d_start   = d_blk * BLOCK_D
    d_offsets = d_start + tl.arange(0, BLOCK_D)
    d_mask    = d_offsets < D

    acc  = tl.zeros([BLOCK_D], dtype=tl.float32)
    base = b_idx * N * D

    for n_start in range(0, N, BLOCK_N):
        n_offsets = n_start + tl.arange(0, BLOCK_N)
        n_mask    = n_offsets < N
        x_off     = base + n_offsets[:, None] * D + d_offsets[None, :]
        mask      = n_mask[:, None] & d_mask[None, :]
        x         = tl.load(x_ptr + x_off, mask=mask, other=0.0)
        acc      += tl.sum(x.to(tl.float32), axis=0)

    acc = acc / N
    out_off = b_idx * D + d_offsets
    tl.store(out_ptr + out_off, acc.to(x_ptr.dtype.element_ty), mask=d_mask)


# ═══════════════════════════════════════════════════════════════════════════════
#  2.  Implicit-GEMM kernel for 1×1 conv  (NCHW input)
# ═══════════════════════════════════════════════════════════════════════════════
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        # BLOCK_N=256 covers N=256 in one tile for better efficiency
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _conv1x1_gemm_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K, HW,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid      = tl.program_id(0)
    grid_m   = tl.cdiv(M, BLOCK_M)
    grid_n   = tl.cdiv(N, BLOCK_N)
    width    = GROUP_M * grid_n
    gid      = pid // width
    gsz      = tl.minimum(grid_m - gid * GROUP_M, GROUP_M)
    pid_m    = gid * GROUP_M + (pid % gsz)
    pid_n    = (pid % width) // gsz

    m_off = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_off = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_msk = m_off < M
    n_msk = n_off < N

    b_m  = m_off // HW
    hw_m = m_off % HW

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for ks in range(0, K, BLOCK_K):
        k_off = ks + tl.arange(0, BLOCK_K)
        k_msk = k_off < K

        # A from NCHW: a[m,k] = a_ptr + b_m*K*HW + k*HW + hw_m
        a_idx = b_m[:, None] * K * HW + k_off[None, :] * HW + hw_m[:, None]
        a     = tl.load(a_ptr + a_idx, mask=m_msk[:, None] & k_msk[None, :], other=0.0)

        # B from weight [N, K]: b[n,k] = b_ptr + n*K + k
        b_idx = n_off[:, None] * K + k_off[None, :]
        b     = tl.load(b_ptr + b_idx, mask=n_msk[:, None] & k_msk[None, :], other=0.0)

        acc  += tl.dot(a, tl.trans(b)).to(tl.float32)

    bias = tl.load(bias_ptr + n_off, mask=n_msk, other=0.0)
    acc  = acc + bias[None, :]

    c_idx = b_m[:, None] * N * HW + n_off[None, :] * HW + hw_m[:, None]
    tl.store(c_ptr + c_idx,
             acc.to(a_ptr.dtype.element_ty),
             mask=m_msk[:, None] & n_msk[None, :])


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified dispatch wrapper  (the single replacement_func for every pass)
# ═══════════════════════════════════════════════════════════════════════════════
@torch.fx.wrap
def dispatch_wrapper(a, b, c, route):
    """
    route == "mean"        : a=x[B,N,D],  b=None, c=None
    route == "conv2d_view" : a=bias[Cout], b=weight[Cout,Cin,1,1], c=x[B,Cin,H,W]
    """
    if route == "mean":
        B   = a.shape[0]
        N   = a.shape[1]
        D   = a.shape[2]
        # Allocate output directly as [B, 1, D] – no .view() needed
        out = torch.empty((B, 1, D), dtype=a.dtype, device=a.device)
        grid = lambda meta: (B * triton.cdiv(D, meta['BLOCK_D']),)
        _mean_neg2_keepdim_kernel[grid](a, out, B, N, D)
        return out

    elif route == "conv2d_view":
        bias   = a
        weight = b
        x      = c
        B      = x.shape[0]
        Cin    = x.shape[1]
        H      = x.shape[2]
        W      = x.shape[3]
        Cout   = weight.shape[0]
        HW     = H * W
        M      = B * HW
        # Pass weight directly – [Cout, Cin, 1, 1] is contiguous, same memory
        # layout as [Cout, Cin]; kernel accesses b_ptr + n*Cin + k which is correct.
        out    = torch.empty((B, Cout, HW), dtype=x.dtype, device=x.device)
        grid   = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(Cout, meta['BLOCK_N']),
        )
        _conv1x1_gemm_kernel[grid](x, weight, out, bias, M, Cout, Cin, HW)
        return out