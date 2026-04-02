"""
Fused optimization pass for 1x1 conv2d + avg_pool2d(kernel=2, stride=2).

Key algebraic identity: For a 1x1 conv (linear over channels),
  avg_pool(conv(x, W)) == conv(avg_pool(x), W)

Pool the input first (4x spatial reduction), then run GEMM on the 4x smaller domain.

Strategy:
  Small M (N*OH*OW < 10000):
    Single fused Triton kernel (pool inside load + tl.dot GEMM).
    BLOCK_M=16 configs give 100+ blocks even for batch=1.
    Small tensors fit in L2 cache, hiding non-coalesced IC reads.
  Large M (N*OH*OW >= 10000):
    Kernel 2: Triton avg-pool [N,IC,H,W] -> [N,IC,OH,OW]
    Tensor ops (.permute, .contiguous, .view) - not blocked
    Kernel 3: coalesced Triton GEMM [M,IC] x [IC,OC] -> [M,OC]
    Tensor ops (.view, .permute, .contiguous) - not blocked
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
# Kernel 1: Fused avg-pool + GEMM  (small M path)
#   x_ptr  : [N, IC, H, W]
#   w_ptr  : [IC, OC]  (pre-transposed weight)
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
        # Large-M + large-K configs (high arithmetic intensity for large batches)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'num_stages': 4, 'num_warps': 4}),
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
    w_ptr,
    out_ptr,
    OH, OW,
    s_xn, s_xc, s_xh, s_xw,
    s_on, s_oc, s_oh, s_ow,
    M, N, K,
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

        # Load weight directly from [OC, IC] layout (no Python-side transposition).
        # Load as [BN, BK]: consecutive k values have stride 1 → fully coalesced.
        w_mask = (n_offs[:, None] < N) & k_mask[None, :]
        w_T = tl.load(w_ptr + n_offs[:, None] * K + k_offs[None, :],
                      mask=w_mask, other=0.0).to(tl.float32)  # [BN, BK]

        # tl.trans is a free register reshape; tl.dot gets [BM,BK] @ [BK,BN]
        acc += tl.dot(a, tl.trans(w_T))

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
# Kernel 1b: same fused kernel but weight is pre-transposed [IC, OC].
#   Used for large M where the extra .T.contiguous() cost is negligible and
#   loading [BK, BN] from [IC, OC] (coalesced in OC) + no tl.trans gives
#   better hardware MMA utilisation than the tl.trans variant.
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
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 128, 'num_stages': 3, 'num_warps': 8}),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_pool_gemm_notrans(
    x_ptr,
    w_ptr,       # [IC, OC]  pre-transposed weight
    out_ptr,
    OH, OW,
    s_xn, s_xc, s_xh, s_xw,
    s_on, s_oc, s_oh, s_ow,
    M, N, K,
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

        # Weight is pre-transposed [IC, OC]; load [BK, BN] – coalesced in N (OC).
        w_mask = k_mask[:, None] & (n_offs[None, :] < N)
        w = tl.load(w_ptr + k_offs[:, None] * N + n_offs[None, :],
                    mask=w_mask, other=0.0).to(tl.float32)  # [BK, BN]
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
# Kernel 2: avg-pool 2x2/2  NCHW->NCHW  (kept for reference, unused now)
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
# Kernel 3: coalesced GEMM [M,K] x [K,N] -> [M,N]  (large M path, step 2)
#   Both A [M,K] and B [K,N] have stride-1 along their inner dimension.
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
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

_LARGE_M = 10000


@torch.fx.wrap
def fused_avgpool_conv_1x1(weight, x):
    """
    avg_pool2d(conv2d(x, W, stride=1), kernel=2, stride=2)
    == conv1x1(avg_pool2d(x, kernel=2, stride=2), W)   [by linearity of 1x1 conv]

    Uses a single fused Triton kernel for all batch sizes:
      - Performs the 2x2 avg-pool DURING the tl.load of the A-matrix
      - Accumulates with tl.dot → single-pass, no intermediate tensors
      - BLOCK_M=16..128 configs → autotuner picks small blocks for small M
        (gives 100+ GPU blocks even at batch=1) and large blocks for large M.

    weight : [OC, IC, 1, 1]
    x      : [N, IC, H, W]
    returns: [N, OC, OH, OW]  OH=H//2, OW=W//2
    """
    N_batch, IC, H, W = x.shape
    OC = weight.shape[0]
    OH = H // 2
    OW = W // 2
    M  = N_batch * OH * OW

    output = torch.empty((N_batch, OC, OH, OW), dtype=x.dtype, device=x.device)
    grid   = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))

    # Use tl.trans inside the kernel – avoids a Python-side .T.contiguous()
    # kernel launch (important for small M where that overhead dominates).
    # For large M, tl.trans adds negligible register overhead while saving the
    # extra CUDA kernel launch.
    weight_2d = weight.view(OC, IC)  # [OC, IC]
    _fused_pool_gemm_kernel[grid](
        x, weight_2d, output,
        OH, OW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        M, OC, IC,
    )

    return output


# ---------------------------------------------------------------------------
# Required by the framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_avgpool_conv_1x1