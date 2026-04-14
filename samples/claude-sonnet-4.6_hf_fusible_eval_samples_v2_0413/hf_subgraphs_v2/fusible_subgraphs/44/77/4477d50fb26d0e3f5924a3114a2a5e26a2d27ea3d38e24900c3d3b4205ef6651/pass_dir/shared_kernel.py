"""
Shared Triton kernel for fused 1x1-Conv + NHWC-reshape + Sigmoid.

All graphs share the structure:
  conv2d(input[N,Cin,H,W], weight[Cout,Cin,1,1], bias[Cout], stride=1, pad=0)
  -> permute(0,2,3,1)           # NCHW -> NHWC
  -> reshape(N, -1, Cout)       # flatten spatial: [N, H*W, Cout]
  -> sigmoid

We fuse everything into one Triton kernel that:
  - reads input in NCHW format
  - computes the 1x1-conv (tiled matmul with bias)
  - applies sigmoid
  - writes output directly in [N, H*W, Cout] layout
"""

import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'Cin', 'Cout'],
)
@triton.jit
def fused_1x1conv_sigmoid_kernel(
    x_ptr,      # [N, Cin, H, W]  (NCHW)
    w_ptr,      # [Cout, Cin]      (2-D view of the 1x1 weight)
    b_ptr,      # [Cout]           (bias)
    o_ptr,      # [M, Cout]        (output, M = N*H*W, contiguous)
    N, Cin, H, W, Cout, M,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)   # tile over M  = N*H*W
    pid_n = tl.program_id(1)   # tile over Cout

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute per-row NCHW spatial coordinates (independent of k loop)
    HW = H * W
    n_batch = m_offs // HW          # batch index
    spatial  = m_offs % HW
    h_idx    = spatial // W
    w_idx    = spatial % W

    for k_start in range(0, Cin, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # ---- Load A tile [BLOCK_M, BLOCK_K] from NCHW input ----
        # input[n, k, h, w] -> offset = n*(Cin*HW) + k*HW + h*W + w
        a_off = (n_batch[:, None] * (Cin * HW)
                 + k_offs[None, :] * HW
                 + h_idx[:, None] * W
                 + w_idx[:, None])
        a_mask = (m_offs[:, None] < M) & (k_offs[None, :] < Cin)
        a = tl.load(x_ptr + a_off, mask=a_mask, other=0.0).to(tl.float32)

        # ---- Load B tile [BLOCK_N, BLOCK_K] from weight[Cout, Cin] ----
        # weight[c_out, k] -> offset = c_out*Cin + k
        b_off = n_offs[:, None] * Cin + k_offs[None, :]
        b_mask = (n_offs[:, None] < Cout) & (k_offs[None, :] < Cin)
        b = tl.load(w_ptr + b_off, mask=b_mask, other=0.0).to(tl.float32)

        # acc[m, n] += sum_k  A[m,k] * B[n,k]   i.e. A @ B^T
        acc = tl.dot(a, tl.trans(b), acc)

    # Add bias + sigmoid
    bias = tl.load(b_ptr + n_offs, mask=n_offs < Cout, other=0.0).to(tl.float32)
    acc  = acc + bias[None, :]
    acc  = tl.sigmoid(acc)

    # Write output  o[m, c_out]  (contiguous [M, Cout])
    o_off  = m_offs[:, None] * Cout + n_offs[None, :]
    o_mask = (m_offs[:, None] < M) & (n_offs[None, :] < Cout)
    tl.store(o_ptr + o_off, acc.to(o_ptr.dtype.element_ty), mask=o_mask)


def compute_conv1x1_sigmoid(bias, weight, x, d0, d2):
    """
    Fused:  conv2d(x, weight, bias, stride=1, pad=0, 1x1)
            -> permute(0,2,3,1)
            -> reshape(d0, -1, d2)
            -> sigmoid

    Returns a tensor of shape (d0, H*W*N//d0, d2).
    Because d0 == N and d2 == Cout in all target graphs,
    the output shape is [N, H*W, Cout].
    """
    N, Cin, H, W = x.shape
    Cout = d2
    M    = N * H * W

    # Allocate flat output [M, Cout]
    o = torch.empty((M, Cout), dtype=x.dtype, device=x.device)

    # View weight as 2-D [Cout, Cin]
    w2d = weight.view(Cout, Cin)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Cout, meta['BLOCK_N']),
    )

    fused_1x1conv_sigmoid_kernel[grid](
        x, w2d, bias, o,
        N, Cin, H, W, Cout, M,
    )

    # Reshape to match the expected output [d0, H*W, d2]
    return o.view(d0, -1, d2)


# ---------------------------------------------------------------------------
# Shared dispatch wrapper – returned by replacement_func() in EVERY pass file.
# Using a route-string as the last argument lets all passes share exactly
# the same replacement_func(), avoiding the output_pass_replacement_func_limit.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def dispatch_conv1x1_sigmoid(in_0, in_1, in_2, route):
    if route == "1_4":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 1, 4)
    elif route == "1_36":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 1, 36)
    elif route == "2_1":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 2, 1)
    elif route == "2_9":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 2, 9)
    elif route == "4_9":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 4, 9)
    elif route == "6_1":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 6, 1)
    elif route == "6_4":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 6, 4)
    elif route == "8_9":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 8, 9)
    elif route == "8_36":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 8, 36)
    elif route == "12_1":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 12, 1)
    elif route == "12_4":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 12, 4)
    elif route == "12_9":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 12, 9)
    elif route == "12_36":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 12, 36)
    elif route == "16_1":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 16, 1)
    elif route == "16_9":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 16, 9)
    elif route == "24_1":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 24, 1)
    elif route == "32_1":
        return compute_conv1x1_sigmoid(in_0, in_1, in_2, 32, 1)
    # fallback (should never be reached)
    return compute_conv1x1_sigmoid(in_0, in_1, in_2, d0=1, d2=1)


# ---------------------------------------------------------------------------
# Universal wrapper (NO route string) – d0 and d2 derived from input shapes.
# All passes share this single function → no replacement_func_limit issues.
# replacement_args always returns (in_0, in_1, in_2) matching pattern args.
# ---------------------------------------------------------------------------
@torch.fx.wrap
def auto_conv1x1_sigmoid(in_0, in_1, in_2):
    """
    Fused 1×1-conv + permute + flatten-spatial + sigmoid.
    d0 = N  (batch size, always equals in_2.shape[0])
    d2 = Cout (output channels, always equals in_0.shape[0])
    Works for ALL (d0, d2) combinations without a route string.
    """
    N    = in_2.shape[0]   # batch size  → d0
    Cout = in_0.shape[0]   # output channels → d2
    return compute_conv1x1_sigmoid(in_0, in_1, in_2, N, Cout)


# ---------------------------------------------------------------------------
# Simpler 3-op fusion: permute(0,2,3,1) + reshape(N,-1,C) + sigmoid
# Input x: [N, C, H, W]  (NCHW conv2d output)
# Output:  [N, H*W, C]   (NHWC flattened + sigmoid applied)
# This avoids fusing conv2d which may have matching difficulties.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def nchw_to_nhwc_sigmoid_kernel(
    x_ptr, o_ptr,
    N, C, H, W,
    N_total,
    BLOCK: tl.constexpr,
):
    """
    NCHW → NHWC conversion with sigmoid.
    Output indexed as flat [N*H*W, C] = M*C elements.
    out[m, c] = sigmoid(x[n, c, h, w]) where m = n*H*W + h*W + w
    """
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    # out_m = spatial index, out_c = channel index
    out_m = offs // C
    out_c = offs % C
    n = out_m // (H * W)
    s = out_m % (H * W)
    h = s // W
    w = s % W
    in_offs = n * (C * H * W) + out_c * (H * W) + h * W + w
    mask = offs < N_total
    xv = tl.load(x_ptr + in_offs, mask=mask, other=0.0).to(tl.float32)
    yv = tl.sigmoid(xv)
    tl.store(o_ptr + offs, yv.to(o_ptr.dtype.element_ty), mask=mask)


@torch.fx.wrap
def auto_permute_reshape_sigmoid(x):
    """
    Fused permute(0,2,3,1) + reshape(N,-1,C) + sigmoid.
    x: [N, C, H, W]  →  returns [N, H*W, C]
    """
    N, C, H, W = x.shape
    M       = N * H * W
    N_total = M * C
    out     = torch.empty((M, C), dtype=x.dtype, device=x.device)
    grid    = lambda meta: (triton.cdiv(N_total, meta['BLOCK']),)
    nchw_to_nhwc_sigmoid_kernel[grid](x, out, N, C, H, W, N_total)
    return out.view(N, H * W, C)


# ---------------------------------------------------------------------------
# Layout-only NCHW→NHWC copy (no sigmoid) – used with permute+reshape pattern.
# The model's original sigmoid is preserved in the graph and runs correctly on
# the contiguous NHWC output, yielding correct results with speedup from the
# fused Triton layout conversion (replaces PyTorch's lazy permute + copy).
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=4),
        triton.Config({'BLOCK': 4096}, num_warps=8),
        triton.Config({'BLOCK': 8192}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def nchw_to_nhwc_copy_kernel(
    x_ptr, o_ptr,
    N, C, H, W,
    N_total,
    BLOCK: tl.constexpr,
):
    """NCHW [N,C,H,W] → NHWC-flat [N*H*W, C] copy (no sigmoid)."""
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    out_m = offs // C
    out_c = offs % C
    n = out_m // (H * W)
    s = out_m % (H * W)
    h = s // W
    w = s % W
    in_offs = n * (C * H * W) + out_c * (H * W) + h * W + w
    mask = offs < N_total
    xv = tl.load(x_ptr + in_offs, mask=mask, other=0.0)
    tl.store(o_ptr + offs, xv, mask=mask)


@torch.fx.wrap
def auto_permute_reshape(x):
    """
    Fused permute(0,2,3,1) + reshape(N,-1,C) – NO sigmoid.
    x: [N, C, H, W] (NCHW) → returns [N, H*W, C] (contiguous NHWC flat).
    Model's original sigmoid still runs on this contiguous output.
    """
    N, C, H, W = x.shape
    M       = N * H * W
    N_total = M * C
    out     = torch.empty((M, C), dtype=x.dtype, device=x.device)
    grid    = lambda meta: (triton.cdiv(N_total, meta['BLOCK']),)
    nchw_to_nhwc_copy_kernel[grid](x, out, N, C, H, W, N_total)
    return out.view(N, H * W, C)