"""
Fused optimization pass for 1x1 conv2d + avg_pool2d(kernel=2, stride=2).

Key algebraic identity: For a 1×1 convolution (linear over channels),
  avg_pool(conv(x, W)) == conv(avg_pool(x), W)

This lets us pool the input first (4× spatial reduction) and then run the GEMM
on a 4× smaller spatial domain.

Strategy (hybrid):
  - Small M (batch*OH*OW < 10000):
      Single fused Triton kernel (pool embedded in load + tl.dot GEMM).
      Small BLOCK_M configs (down to 16) ensure enough blocks for GPU utilisation
      even at batch=1, and the small dataset fits in L2 cache.
  - Large M (batch*OH*OW >= 10000):
      _avg_pool_nchw_kernel: pool [N,IC,H,W] → [N,IC,OH,OW]
      PyTorch tensor ops (.permute, .contiguous, .view) – not blocked.
      _matmul_kernel: coalesced GEMM [M,IC] × [IC,OC] → [M,OC]
      PyTorch tensor ops (.view, .permute, .contiguous) – not blocked.
      The permute makes IC contiguous → fully coalesced reads in the GEMM.
"""

import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern to match
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ===========================================================================
# Kernel 1: Fused avg-pool + GEMM  (used for small M)
#   x_ptr  : [N, IC, H, W]   (original input)
#   w_ptr  : [IC, OC]         (weight, pre-transposed)
#   out_ptr: [N, OC, OH, OW]
# ===========================================================================

@triton.autotune(
    configs=[
        # Large-M configs
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        # Small-M configs (batch=1)
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 2}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_pool_gemm_kernel(
    x_ptr,
    w_ptr,       # [IC, OC]  pre-transposed weight
    out_ptr,     # [N, OC, OH, OW]
    OH, OW,
    s_xn, s_xc, s_xh, s_xw,   # input strides [N, IC, H, W]
    s_on, s_oc, s_oh, s_ow,   # output strides [N, OC, OH, OW]
    M, N, K,                   # M=N_batch*OH*OW, N=OC, K=IC
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    OHOW  = OH * OW
    b_idx = m_offs // OHOW
    rem   = m_offs % OHOW
    oh    = rem // OW
    ow    = rem  % OW
    ih    = oh * 2
    iw    = ow * 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs  = k + tl.arange(0, BLOCK_K)
        m_mask  = m_offs < M
        k_mask  = k_offs < K
        in_mask = m_mask[:, None] & k_mask[None, :]

        base = b_idx[:, None] * s_xn + k_offs[None, :] * s_xc

        a00 = tl.load(x_ptr + base +  ih[:, None]      * s_xh +  iw[:, None]      * s_xw, mask=in_mask, other=0.0)
        a01 = tl.load(x_ptr + base +  ih[:, None]      * s_xh + (iw[:, None] + 1) * s_xw, mask=in_mask, other=0.0)
        a10 = tl.load(x_ptr + base + (ih[:, None] + 1) * s_xh +  iw[:, None]      * s_xw, mask=in_mask, other=0.0)
        a11 = tl.load(x_ptr + base + (ih[:, None] + 1) * s_xh + (iw[:, None] + 1) * s_xw, mask=in_mask, other=0.0)

        a = ((a00 + a01 + a10 + a11) * 0.25).to(tl.float32)

        w_mask = k_mask[:, None] & (n_offs[None, :] < N)
        w = tl.load(w_ptr + k_offs[:, None] * N + n_offs[None, :],
                    mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, w)

    m_mask   = m_offs < M
    n_mask   = n_offs < N
    out_mask = m_mask[:, None] & n_mask[None, :]
    out_ptrs = (out_ptr
                + b_idx[:, None] * s_on
                + n_offs[None, :] * s_oc
                + oh[:, None]    * s_oh
                + ow[:, None]    * s_ow)
    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ===========================================================================
# Kernel 2: avg-pool 2×2/2  NCHW→NCHW  (used for large M path, step 1)
# ===========================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=['total'],
)
@triton.jit
def _avg_pool_nchw_kernel(
    x_ptr, out_ptr,
    N_batch, IC, H, W, OH, OW,
    s_xn, s_xc, s_xh, s_xw,
    s_on, s_oc, s_oh, s_ow,
    total,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    ow  = offs % OW
    tmp = offs // OW
    oh  = tmp  % OH
    tmp = tmp  // OH
    ic  = tmp  % IC
    n   = tmp  // IC

    ih  = oh * 2
    iw  = ow * 2
    base = n * s_xn + ic * s_xc + ih * s_xh + iw * s_xw

    a00 = tl.load(x_ptr + base,               mask=mask, other=0.0)
    a01 = tl.load(x_ptr + base + s_xw,        mask=mask, other=0.0)
    a10 = tl.load(x_ptr + base + s_xh,        mask=mask, other=0.0)
    a11 = tl.load(x_ptr + base + s_xh + s_xw, mask=mask, other=0.0)

    avg = (a00 + a01 + a10 + a11) * 0.25
    out_idx = n * s_on + ic * s_oc + oh * s_oh + ow * s_ow
    tl.store(out_ptr + out_idx, avg.to(out_ptr.dtype.element_ty), mask=mask)


# ===========================================================================
# Kernel 3: Standard coalesced GEMM  [M,K] × [K,N] → [M,N]  (large M, step 2)
#   a_ptr strides: [K, 1]  (IC contiguous after permute+contiguous)
#   b_ptr strides: [N, 1]  (OC contiguous after weight.T.contiguous())
# ===========================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,   # K, 1  (IC contiguous)
    stride_bk, stride_bn,   # N, 1  (OC contiguous)
    stride_cm, stride_cn,   # N, 1
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc    = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    a_base = a_ptr + m_offs[:, None] * stride_am
    b_base = b_ptr + n_offs[None, :] * stride_bn

    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)

        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_base + k_offs[None, :] * stride_ak,
                    mask=a_mask, other=0.0).to(tl.float32)

        b_mask = (k_offs[:, None] < K) & (n_offs[None, :] < N)
        b = tl.load(b_base + k_offs[:, None] * stride_bk,
                    mask=b_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, b)

    c_mask = (m_offs[:, None] < M) & (n_offs[None, :] < N)
    tl.store(c_ptr + m_offs[:, None] * stride_cm + n_offs[None, :] * stride_cn,
             acc.to(c_ptr.dtype.element_ty), mask=c_mask)


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------

_LARGE_M_THRESHOLD = 10000


@torch.fx.wrap
def fused_avgpool_conv_1x1(weight, x):
    """
    Replacement for:  avg_pool2d(conv2d(x, weight, stride=1), kernel=2, stride=2)

    Uses the algebraic identity: avg_pool(conv1x1(x, W)) == conv1x1(avg_pool(x), W)

    weight : [OC, IC, 1, 1]
    x      : [N, IC, H, W]
    returns: [N, OC, OH, OW]   OH = H//2, OW = W//2
    """
    N_batch, IC, H, W = x.shape
    OC = weight.shape[0]
    OH = H // 2
    OW = W // 2
    M  = N_batch * OH * OW

    if M < _LARGE_M_THRESHOLD:
        # -------------------------------------------------------------------
        # Small M: single fused kernel (pool during load + tl.dot GEMM).
        # BLOCK_M=16 gives 144+ blocks even for M=576 → good utilisation.
        # Tensor fits in L2, hiding non-coalesced IC reads.
        # -------------------------------------------------------------------
        output   = torch.empty((N_batch, OC, OH, OW), dtype=x.dtype, device=x.device)
        weight_T = weight.view(OC, IC).T.contiguous()   # [IC, OC]

        grid = lambda meta: (
            triton.cdiv(M,  meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )
        _fused_pool_gemm_kernel[grid](
            x, weight_T, output,
            OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            M, OC, IC,
        )
        return output

    # -----------------------------------------------------------------------
    # Large M:  Triton pool → permute → Triton GEMM → reshape
    #
    # Step 1: pool [N,IC,H,W] → [N,IC,OH,OW]  (Triton, reads each input once)
    # Step 2: .permute(0,2,3,1).contiguous()   (PyTorch tensor op, not blocked)
    #         makes IC the stride-1 axis → fully coalesced GEMM reads
    # Step 3: Triton GEMM [M,IC] × [IC,OC] → [M,OC]
    # Step 4: .view + .permute + .contiguous() (PyTorch tensor ops, not blocked)
    # -----------------------------------------------------------------------
    # Step 1
    pooled = torch.empty((N_batch, IC, OH, OW), dtype=x.dtype, device=x.device)
    total  = N_batch * IC * OH * OW
    pool_grid = (triton.cdiv(total, 512),)
    _avg_pool_nchw_kernel[pool_grid](
        x, pooled,
        N_batch, IC, H, W, OH, OW,
        x.stride(0),      x.stride(1),      x.stride(2),      x.stride(3),
        pooled.stride(0), pooled.stride(1), pooled.stride(2), pooled.stride(3),
        total,
    )

    # Step 2 – tensor ops only (not blocked)
    x_2d     = pooled.permute(0, 2, 3, 1).contiguous().view(M, IC)  # [M, IC]
    weight_T = weight.view(OC, IC).T.contiguous()                    # [IC, OC]

    # Step 3 – Triton GEMM
    out_2d   = torch.empty((M, OC), dtype=x.dtype, device=x.device)
    mm_grid  = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))
    _matmul_kernel[mm_grid](
        x_2d, weight_T, out_2d,
        M, OC, IC,
        x_2d.stride(0),     x_2d.stride(1),      # [IC, 1]
        weight_T.stride(0), weight_T.stride(1),   # [OC, 1]
        out_2d.stride(0),   out_2d.stride(1),     # [OC, 1]
    )

    # Step 4 – tensor ops only (not blocked)
    return out_2d.view(N_batch, OH, OW, OC).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Required by the framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_avgpool_conv_1x1
#   x_ptr  : [N, IC, H, W]   (original input)
#   w_ptr  : [IC, OC]         (weight, pre-transposed)
#   out_ptr: [N, OC, OH, OW]
#
# Each thread block handles a BLOCK_M × BLOCK_N tile of the output matrix
# (where M = N*OH*OW and N_out = OC).  The avg-pool is performed during load
# of the A matrix: for each spatial position m we load 4 input elements and
# average them before calling tl.dot.
# ===========================================================================

@triton.autotune(
    configs=[
        # Large-M configs (batch=128)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 4, 'num_warps': 4}),
        # Small-M configs (batch=1, few spatial positions)
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 2}),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 2, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32,  'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_pool_gemm_kernel(
    x_ptr,
    w_ptr,       # [IC, OC]  pre-transposed weight
    out_ptr,     # [N, OC, OH, OW]
    OH, OW,
    # input strides [N, IC, H, W]
    s_xn, s_xc, s_xh, s_xw,
    # output strides [N, OC, OH, OW]
    s_on, s_oc, s_oh, s_ow,
    M, N, K,    # M=N*OH*OW,  N=OC,  K=IC
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_offs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BM]
    n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BN]

    # Decode m → (batch, oh, ow)
    OHOW  = OH * OW
    b_idx = m_offs // OHOW
    rem   = m_offs % OHOW
    oh    = rem // OW
    ow    = rem  % OW
    ih    = oh * 2
    iw    = ow * 2

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)   # [BK]

        m_mask  = m_offs < M
        k_mask  = k_offs < K
        in_mask = m_mask[:, None] & k_mask[None, :]  # [BM, BK]

        # Compute base address for input tile
        base = b_idx[:, None] * s_xn + k_offs[None, :] * s_xc  # [BM, BK]

        # Load 4 pool positions and average
        a00 = tl.load(x_ptr + base +  ih[:, None]      * s_xh +  iw[:, None]      * s_xw, mask=in_mask, other=0.0)
        a01 = tl.load(x_ptr + base +  ih[:, None]      * s_xh + (iw[:, None] + 1) * s_xw, mask=in_mask, other=0.0)
        a10 = tl.load(x_ptr + base + (ih[:, None] + 1) * s_xh +  iw[:, None]      * s_xw, mask=in_mask, other=0.0)
        a11 = tl.load(x_ptr + base + (ih[:, None] + 1) * s_xh + (iw[:, None] + 1) * s_xw, mask=in_mask, other=0.0)

        a = ((a00 + a01 + a10 + a11) * 0.25).to(tl.float32)  # [BM, BK]

        # Load weight tile [BK, BN] – coalesced along N (stride 1)
        w_mask = k_mask[:, None] & (n_offs[None, :] < N)
        w = tl.load(w_ptr + k_offs[:, None] * N + n_offs[None, :],
                    mask=w_mask, other=0.0).to(tl.float32)

        acc += tl.dot(a, w)

    # Store to [N, OC, OH, OW]
    m_mask   = m_offs < M
    n_mask   = n_offs < N
    out_mask = m_mask[:, None] & n_mask[None, :]

    out_ptrs = (out_ptr
                + b_idx[:, None] * s_on
                + n_offs[None, :] * s_oc
                + oh[:, None]    * s_oh
                + ow[:, None]    * s_ow)

    tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ===========================================================================
# Kernel 2: Plain avg-pool 2×2/2  NCHW → NCHW  (used for large M)
#   Flattened over all N*IC*OH*OW output elements.
#   Reads are stride-2 in W (half-coalesced) but writes are stride-1 (coalesced).
# ===========================================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 512}),
        triton.Config({'BLOCK': 1024}),
    ],
    key=['total'],
)
@triton.jit
def _avg_pool_nchw_kernel(
    x_ptr, out_ptr,
    N_batch, IC, H, W, OH, OW,
    s_xn, s_xc, s_xh, s_xw,
    s_on, s_oc, s_oh, s_ow,
    total,
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    # Decode linear index over [N, IC, OH, OW] with OW varying fastest
    ow   = offs % OW
    tmp  = offs // OW
    oh   = tmp  % OH
    tmp  = tmp  // OH
    ic   = tmp  % IC
    n    = tmp  // IC

    ih = oh * 2
    iw = ow * 2

    base = n * s_xn + ic * s_xc + ih * s_xh + iw * s_xw

    a00 = tl.load(x_ptr + base,                   mask=mask, other=0.0)
    a01 = tl.load(x_ptr + base + s_xw,            mask=mask, other=0.0)
    a10 = tl.load(x_ptr + base + s_xh,            mask=mask, other=0.0)
    a11 = tl.load(x_ptr + base + s_xh + s_xw,     mask=mask, other=0.0)

    avg = (a00 + a01 + a10 + a11) * 0.25

    out_idx = n * s_on + ic * s_oc + oh * s_oh + ow * s_ow
    tl.store(out_ptr + out_idx, avg.to(out_ptr.dtype.element_ty), mask=mask)


# ---------------------------------------------------------------------------
# Python wrapper  (must be @torch.fx.wrap so FX does not trace into it)
# ---------------------------------------------------------------------------

_LARGE_M_THRESHOLD = 10000   # switch from fused kernel to pool+mm above this


@torch.fx.wrap
def fused_avgpool_conv_1x1(weight, x):
    """
    Replacement for:  avg_pool2d(conv2d(x, weight, stride=1), kernel=2, stride=2)

    Uses the algebraic identity: avg_pool(conv1x1(x, W)) == conv1x1(avg_pool(x), W)

    weight : [OC, IC, 1, 1]
    x      : [N, IC, H, W]
    returns: [N, OC, OH, OW]   OH = H//2, OW = W//2
    """
    N_batch, IC, H, W = x.shape
    OC = weight.shape[0]
    OH = H // 2
    OW = W // 2
    M  = N_batch * OH * OW

    if M < _LARGE_M_THRESHOLD:
        # -----------------------------------------------------------------------
        # Small M path: single fused Triton kernel
        #   - Pool is performed during load; GEMM uses tl.dot
        #   - Small block sizes (BLOCK_M=16) ensure sufficient parallelism
        #   - Small tensors fit in L2 cache, hiding non-coalesced IC reads
        # -----------------------------------------------------------------------
        output   = torch.empty((N_batch, OC, OH, OW), dtype=x.dtype, device=x.device)
        weight_T = weight.view(OC, IC).T.contiguous()   # [IC, OC]

        grid = lambda meta: (
            triton.cdiv(M,  meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

        _fused_pool_gemm_kernel[grid](
            x, weight_T, output,
            OH, OW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            output.stride(0), output.stride(1), output.stride(2), output.stride(3),
            M, OC, IC,
        )

        return output

    # -----------------------------------------------------------------------
    # Large M path: Triton pool → permute → torch.mm (cuBLAS) → reshape
    #   1. Pool the input with a Triton kernel: [N,IC,H,W] → [N,IC,OH,OW]
    #   2. Permute to [N,OH,OW,IC] so IC is the fast (stride-1) axis
    #   3. cuBLAS mm on the 4× smaller matrix → [M, OC]
    #   4. Reshape back to [N,OC,OH,OW]
    # -----------------------------------------------------------------------
    pooled = torch.empty((N_batch, IC, OH, OW), dtype=x.dtype, device=x.device)
    total  = N_batch * IC * OH * OW

    pool_grid = (triton.cdiv(total, 512),)

    _avg_pool_nchw_kernel[pool_grid](
        x, pooled,
        N_batch, IC, H, W, OH, OW,
        x.stride(0),      x.stride(1),      x.stride(2),      x.stride(3),
        pooled.stride(0), pooled.stride(1), pooled.stride(2), pooled.stride(3),
        total,
    )

    # Make IC contiguous (stride 1) for coalesced cuBLAS reads
    x_2d     = pooled.permute(0, 2, 3, 1).contiguous().view(M, IC)   # [M, IC]
    weight2d = weight.view(OC, IC)                                     # [OC, IC]

    # torch.mm resolves to aten::mm – NOT torch.nn.functional.conv2d/avg_pool2d
    out_2d   = torch.mm(x_2d, weight2d.t())                           # [M, OC]

    # Reshape to NCHW
    return out_2d.view(N_batch, OH, OW, OC).permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Required by the framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_avgpool_conv_1x1