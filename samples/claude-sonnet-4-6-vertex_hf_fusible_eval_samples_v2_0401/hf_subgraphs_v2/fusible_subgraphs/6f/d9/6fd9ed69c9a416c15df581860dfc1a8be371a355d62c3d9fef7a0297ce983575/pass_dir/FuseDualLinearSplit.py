"""
Pass: FuseDualLinearSplit
Fuses the ENTIRE forward():
  - linear1(in_5, in_1, in_0)  + slice first/second halves + view + unsqueeze
  - in_4.reshape + linear2     + slice first/second halves

Strategy:
  1. Use tensor-method matmul (x @ w.t(), cuBLAS) for the heavy GEMM compute.
  2. Use a fast Triton kernel only for the bias-add + split post-processing.
     This kernel reads the [M,N] GEMM output (hot in L2), adds bias, and
     writes two contiguous [M, N//2] halves – all in one pass.
  A single combined Triton kernel covers both GEMMs via a 3-D grid
  (pid_b selects which GEMM's output to process), reducing kernel-launch
  overhead and improving SM utilisation compared to two separate splits.
"""
import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'N'],
)
@triton.jit
def _dual_bias_split_kernel(
    # GEMM outputs (already computed, hot in L2)
    y1_ptr, b1_ptr,            # GEMM-1: [M,N] output, [N] bias
    y2_ptr, b2_ptr,            # GEMM-2: [M,N] output, [N] bias
    # Four contiguous output buffers [M, half_N] each
    out1a_ptr, out1b_ptr,      # GEMM-1 first/second halves
    out2a_ptr, out2b_ptr,      # GEMM-2 first/second halves
    M, N, half_N,
    sy_m,                      # y row stride (= N for contiguous)
    so_m,                      # output row stride (= half_N)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """
    Elementwise kernel: bias_add + split.
    3-D grid: (M-tiles, N-tiles, 2-batch).
    pid_b=0 → GEMM-1,  pid_b=1 → GEMM-2.
    Loads both y1 and y2 tiles (both hot in L2), selects with tl.where.
    No inner K-loop – pure memory-bandwidth bound.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    m_mask = offs_m[:, None] < M
    n_mask = offs_n[None, :] < N

    # Load both GEMM outputs (both cached in L2 after cuBLAS writes)
    y1 = tl.load(y1_ptr + offs_m[:, None] * sy_m + offs_n[None, :],
                 mask=m_mask & n_mask, other=0.0)
    y2 = tl.load(y2_ptr + offs_m[:, None] * sy_m + offs_n[None, :],
                 mask=m_mask & n_mask, other=0.0)

    b1 = tl.load(b1_ptr + offs_n, mask=offs_n < N, other=0.0)
    b2 = tl.load(b2_ptr + offs_n, mask=offs_n < N, other=0.0)

    use1 = (pid_b == 0)
    y  = tl.where(use1, y1, y2) + tl.where(use1, b1, b2)[None, :]

    # First-half columns (0 .. half_N-1)
    first_mask  = m_mask & (offs_n[None, :] < half_N)
    a_offs      = offs_m[:, None] * so_m + offs_n[None, :]

    # Second-half columns (half_N .. N-1)
    second_mask = m_mask & (offs_n[None, :] >= half_N) & n_mask
    b_offs      = offs_m[:, None] * so_m + (offs_n[None, :] - half_N)

    tl.store(out1a_ptr + a_offs, y, mask=first_mask  &  use1)
    tl.store(out1b_ptr + b_offs, y, mask=second_mask &  use1)
    tl.store(out2a_ptr + a_offs, y, mask=first_mask  & ~use1)
    tl.store(out2b_ptr + b_offs, y, mask=second_mask & ~use1)


# ── wrapped inner launcher (leaf in FX graph) ────────────────────────────────

@torch.fx.wrap
def _run_dual_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0: bias1 [N],  in_1: weight1 [N,K],  in_5: x1 [M,K]
    in_2: bias2 [N],  in_3: weight2 [N,K],  in_4: [1,150,1,512] → x2 [M,K]

    Steps:
      1. GEMMs via tensor-matmul operator (dispatches to cuBLAS).
      2. Triton fused bias-add + split for both outputs in one kernel call.

    Returns: (out1a, out1b, out2a, out2b)  each [M, half_N]
    """
    x1 = in_5                              # [M, K]
    x2 = in_4.reshape(-1, in_1.shape[1])  # [M, K]

    # Compute GEMMs – uses cuBLAS via the tensor @ operator
    y1 = x1 @ in_1.t()   # [M, N]  (no bias yet)
    y2 = x2 @ in_3.t()   # [M, N]  (no bias yet)

    M      = y1.shape[0]
    N      = y1.shape[1]
    half_N = N // 2

    out1a = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)
    out1b = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)
    out2a = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)
    out2b = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)

    # Single Triton kernel: bias-add + split for both GEMMs (3-D grid)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        2,
    )

    _dual_bias_split_kernel[grid](
        y1, in_0, y2, in_2,
        out1a, out1b, out2a, out2b,
        M, N, half_N,
        y1.stride(0),   # sy_m = N  (contiguous)
        half_N,         # so_m = half_N  (contiguous output)
    )

    return (out1a, out1b, out2a, out2b)


# ── replacement (NOT @torch.fx.wrap → FX traces, produces 4 return nodes) ───

def fused_dual_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Maps outputs to pattern return order (tmp_11, tmp_12, tmp_8, tmp_13):
      tmp_8  = out1b                     [M, half_N]
      tmp_13 = out1a.unsqueeze(-2)       [M, 1, half_N]
      tmp_11 = out2a.reshape(300,-1,256) [300, 1, 256]
      tmp_12 = out2b.reshape(300,-1,256) [300, 1, 256]
    """
    result = _run_dual_linear(in_0, in_1, in_2, in_3, in_4, in_5)
    out1a  = result[0]
    out1b  = result[1]
    out2a  = result[2]
    out2b  = result[3]

    tmp_8  = out1b
    tmp_13 = out1a.unsqueeze(-2)
    tmp_11 = out2a.reshape(300, -1, 256)
    tmp_12 = out2b.reshape(300, -1, 256)

    return (tmp_11, tmp_12, tmp_8, tmp_13)


# ── pattern, replacement_args, replacement_func ─────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    # First linear path
    tmp_4  = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5  = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6  = tmp_5.view(-1, 256)
    tmp_7  = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8  = tmp_7.view(-1, 256)
    # Second linear path
    tmp_9  = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    # Combine
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_11, tmp_12, tmp_8, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_dual_linear


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 3, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 3, 'num_warps': 4}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32, 'num_stages': 4, 'num_warps': 8}),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64, 'num_stages': 4, 'num_warps': 8}),
    ],
    key=['M', 'K', 'N'],
)
@triton.jit
def _dual_linear_split_kernel(
    # GEMM-1 inputs: x1 [M,K], w1 [N,K], b1 [N]
    x1_ptr, w1_ptr, b1_ptr,
    # GEMM-2 inputs: x2 [M,K], w2 [N,K], b2 [N]
    x2_ptr, w2_ptr, b2_ptr,
    # Four output buffers [M, half_N] each
    out1a_ptr, out1b_ptr,    # GEMM-1: first half, second half
    out2a_ptr, out2b_ptr,    # GEMM-2: first half, second half
    M, K, N, half_N,
    # GEMM-1 input strides
    sx1m, sx1k, sw1n, sw1k,
    # GEMM-2 input strides
    sx2m, sx2k, sw2n, sw2k,
    # Output row stride (same for all 4 outputs = half_N)
    so_m,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    3-D grid: (pid_m, pid_n, pid_b).
    pid_b=0 → GEMM-1,  pid_b=1 → GEMM-2.
    Both input tiles are loaded; tl.where selects the active one.
    Since all threads in a block share the same pid_b, there is NO warp divergence.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)   # 0 or 1 – scalar

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Tile-pointer initialisation for both GEMMs
    x1_ptrs = x1_ptr + offs_m[:, None] * sx1m + offs_k[None, :] * sx1k
    x2_ptrs = x2_ptr + offs_m[:, None] * sx2m + offs_k[None, :] * sx2k
    w1_ptrs = w1_ptr + offs_n[None, :] * sw1n + offs_k[:, None] * sw1k
    w2_ptrs = w2_ptr + offs_n[None, :] * sw2n + offs_k[:, None] * sw2k

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Scalar condition (no warp divergence since pid_b is block-uniform)
    use1 = (pid_b == 0)

    for k in range(tl.cdiv(K, BLOCK_K)):
        k_curr = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_curr[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_curr[:, None] < K)

        # Load both – extra bandwidth negligible for compute-bound tiles
        a1 = tl.load(x1_ptrs, mask=a_mask, other=0.0)
        a2 = tl.load(x2_ptrs, mask=a_mask, other=0.0)
        b1 = tl.load(w1_ptrs, mask=b_mask, other=0.0)
        b2 = tl.load(w2_ptrs, mask=b_mask, other=0.0)

        # Select active GEMM
        a = tl.where(use1, a1, a2)
        b = tl.where(use1, b1, b2)
        acc = tl.dot(a, b, acc)

        x1_ptrs += BLOCK_K * sx1k
        x2_ptrs += BLOCK_K * sx2k
        w1_ptrs += BLOCK_K * sw1k
        w2_ptrs += BLOCK_K * sw2k

    # Add bias (load both, select)
    bias1 = tl.load(b1_ptr + offs_n, mask=offs_n < N, other=0.0)
    bias2 = tl.load(b2_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc   = acc + tl.where(use1, bias1, bias2)[None, :]

    m_mask = offs_m[:, None] < M

    # First-half columns (0 .. half_N-1)
    first_mask  = m_mask & (offs_n[None, :] < half_N)
    a_offs      = offs_m[:, None] * so_m + offs_n[None, :]

    # Second-half columns (half_N .. N-1)
    second_mask = m_mask & (offs_n[None, :] >= half_N) & (offs_n[None, :] < N)
    b_offs      = offs_m[:, None] * so_m + (offs_n[None, :] - half_N)

    # Masked stores – inactive batch gets mask=False → no write
    tl.store(out1a_ptr + a_offs, acc, mask=first_mask  & use1)
    tl.store(out1b_ptr + b_offs, acc, mask=second_mask & use1)
    tl.store(out2a_ptr + a_offs, acc, mask=first_mask  & ~use1)
    tl.store(out2b_ptr + b_offs, acc, mask=second_mask & ~use1)


# ── wrapped inner launcher (leaf in FX graph) ────────────────────────────────

@torch.fx.wrap
def _run_dual_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    in_0: bias1 [N],  in_1: weight1 [N,K],  in_5: x1 [M,K]
    in_2: bias2 [N],  in_3: weight2 [N,K],  in_4: [1,150,1,512] → x2 [M,K]
    Returns: (out1a, out1b, out2a, out2b)  each [M, half_N]
    """
    x1 = in_5                             # [M, K]
    x2 = in_4.reshape(-1, in_1.shape[1]) # [M, K]

    M      = x1.shape[0]
    K      = x1.shape[1]
    N      = in_1.shape[0]
    half_N = N // 2

    out1a = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)
    out1b = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)
    out2a = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)
    out2b = torch.empty((M, half_N), dtype=x1.dtype, device=x1.device)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        2,                                # batch dimension
    )

    _dual_linear_split_kernel[grid](
        x1, in_1, in_0,          # GEMM-1 inputs
        x2, in_3, in_2,          # GEMM-2 inputs
        out1a, out1b,             # GEMM-1 outputs
        out2a, out2b,             # GEMM-2 outputs
        M, K, N, half_N,
        x1.stride(0),  x1.stride(1),
        in_1.stride(0), in_1.stride(1),
        x2.stride(0),  x2.stride(1),
        in_3.stride(0), in_3.stride(1),
        half_N,                           # so_m = row stride of each output
    )

    return (out1a, out1b, out2a, out2b)


# ── replacement (NOT @torch.fx.wrap → FX traces this → 4 separate return nodes) ──

def fused_dual_linear(in_0, in_1, in_2, in_3, in_4, in_5):
    """
    Maps kernel outputs to pattern return order (tmp_11, tmp_12, tmp_8, tmp_13):
      tmp_8  = out1b                   [M, half_N]
      tmp_13 = out1a.unsqueeze(-2)     [M, 1, half_N]
      tmp_11 = out2a.reshape(300,-1,256) [300, 1, 256]
      tmp_12 = out2b.reshape(300,-1,256) [300, 1, 256]
    """
    result = _run_dual_linear(in_0, in_1, in_2, in_3, in_4, in_5)
    out1a  = result[0]   # [M, half_N] – GEMM-1 first half
    out1b  = result[1]   # [M, half_N] – GEMM-1 second half  (= tmp_8)
    out2a  = result[2]   # [M, half_N] – GEMM-2 first half
    out2b  = result[3]   # [M, half_N] – GEMM-2 second half

    tmp_8  = out1b
    tmp_13 = out1a.unsqueeze(-2)          # [M, 1, half_N]
    tmp_11 = out2a.reshape(300, -1, 256)  # [300, 1, 256]
    tmp_12 = out2b.reshape(300, -1, 256)  # [300, 1, 256]

    return (tmp_11, tmp_12, tmp_8, tmp_13)


# ── pattern, replacement_args, replacement_func ─────────────────────────────

def pattern(in_0, in_1, in_2, in_3, in_4, in_5):
    # First linear path
    tmp_4  = torch.nn.functional.linear(in_5, in_1, in_0)
    tmp_5  = tmp_4[slice(None, None, None), slice(None, 256, None)]
    tmp_6  = tmp_5.view(-1, 256)
    tmp_7  = tmp_4[slice(None, None, None), slice(-256, None, None)]
    tmp_8  = tmp_7.view(-1, 256)
    # Second linear path
    tmp_9  = in_4.reshape(300, -1, 256)
    tmp_10 = torch.nn.functional.linear(tmp_9, in_3, in_2)
    tmp_11 = tmp_10[Ellipsis, slice(None, 256, None)]
    tmp_12 = tmp_10[Ellipsis, slice(-256, None, None)]
    # Combine
    tmp_13 = tmp_6.unsqueeze(-2)
    return (tmp_11, tmp_12, tmp_8, tmp_13)


def replacement_args(in_0, in_1, in_2, in_3, in_4, in_5):
    return (in_0, in_1, in_2, in_3, in_4, in_5)


def replacement_func():
    return fused_dual_linear