"""
Optimize conv2d(1x1) + avg_pool2d(2x2, stride=2) via linearity reordering.

Math: avg_pool(conv(x, w)) = conv(avg_pool(x), w)  [both are linear]

Two-step Triton implementation (no torch.conv2d, no permute):
  1. Triton avg_pool2d:  [N, C_in, H, W] → [N, C_in, H/2, W/2]  (NCHW, simple)
  2. Triton GEMM:        pooled + weight → [N, C_out, H/2, W/2]  (NCHW)

GEMM memory coalescing tricks (NCHW layout):
  - A = pooled [N, C_in, H_out, W_out]: A[m,k] has stride H_out*W_out for k.
    → Load A as TRANSPOSED tile [BLOCK_K, BLOCK_M] where m_spatial (last dim) 
      has stride 1 → coalesced for consecutive m (spatial) values.
  - B = weight [C_out, C_in]: B[n,k] has stride 1 for k.
    → Load as [BLOCK_N, BLOCK_K] → coalesced.
  - Output [N, C_out, H_out, W_out]: same NCHW layout as A.
    → Store TRANSPOSED tile [BLOCK_N, BLOCK_M] with m_spatial last → coalesced.

Use native dtype in tl.dot for tensor-core utilization (fp16/bf16/TF32).
"""

import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# Pattern to match
# ─────────────────────────────────────────────────────────────────────────────

def pattern(in_0, in_1):
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: 2×2 avg_pool2d (stride 2) in NCHW layout
# Sequential output offsets → fully coalesced writes.
# Input reads: w varies fastest → nearly coalesced per warp.
# ─────────────────────────────────────────────────────────────────────────────

@triton.jit
def _avg_pool2x2_nchw(
    inp_ptr, out_ptr,
    H_in, W_in,
    H_out, W_out,
    total,              # N * C * H_out * W_out
    BLOCK: tl.constexpr,
):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total

    w_out = offs % W_out
    h_out = (offs // W_out) % H_out
    nc    = offs // (H_out * W_out)

    h0 = h_out * 2;  h1 = h0 + 1
    w0 = w_out * 2;  w1 = w0 + 1
    base = nc * (H_in * W_in)

    v00 = tl.load(inp_ptr + base + h0 * W_in + w0, mask=mask, other=0.0).to(tl.float32)
    v10 = tl.load(inp_ptr + base + h1 * W_in + w0, mask=mask, other=0.0).to(tl.float32)
    v01 = tl.load(inp_ptr + base + h0 * W_in + w1, mask=mask, other=0.0).to(tl.float32)
    v11 = tl.load(inp_ptr + base + h1 * W_in + w1, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + offs, ((v00 + v10 + v01 + v11) * 0.25).to(inp_ptr.dtype.element_ty), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: 1×1 conv as GEMM on NCHW pooled tensor (no permute)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # BLOCK_N=256 = full N dimension → A read only ONCE (no N-tiling reuse overhead)
        # Large M: BLOCK_M=128, 3 pipeline stages (fits float32: A=16KB+B=32KB=48KB×3=144KB)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        # Medium M: BLOCK_M=64, 4 pipeline stages (40KB×4=160KB < 164KB limit)
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Small-medium M: more tiles → better SM occupancy
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        # Very small M: maximize tile count
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N_size', 'K'],
)
@triton.jit
def _conv1x1_nchw_gemm(
    a_ptr,   # NCHW pooled  [N_batch, C_in,  H_out, W_out]
    b_ptr,   # weight       [C_out,   C_in]
    c_ptr,   # NCHW output  [N_batch, C_out, H_out, W_out]
    M,       # N_batch * H_out * W_out
    K,       # C_in
    N_size,  # C_out
    HW_out,  # H_out * W_out
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    n_offs = n_start + tl.arange(0, BLOCK_N)   # [BLOCK_N]

    m_mask = m_offs < M
    n_mask = n_offs < N_size

    # Decode m = b * HW_out + spatial
    b_idx     = m_offs // HW_out    # batch index     [BLOCK_M]
    m_spatial = m_offs % HW_out     # h*W_out + w     [BLOCK_M]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # ── A tile [BLOCK_K, BLOCK_M] transposed ──────────────────────────
        # A[m, k] = a_ptr + b_idx * K * HW_out + k * HW_out + m_spatial
        # Last dim = m_spatial (stride 1) → COALESCED
        a_mask_T = k_mask[:, None] & m_mask[None, :]
        a_tile_T = tl.load(
            a_ptr + b_idx[None, :] * (K * HW_out)
                  + k_offs[:, None] * HW_out
                  + m_spatial[None, :],
            mask=a_mask_T, other=0.0
        )   # [BLOCK_K, BLOCK_M], native dtype

        # ── B tile [BLOCK_N, BLOCK_K] ─────────────────────────────────────
        # B[n, k] = weight[n, k] = b_ptr + n * K + k
        # Last dim = k (stride 1) → COALESCED
        b_mask = n_mask[:, None] & k_mask[None, :]
        b_tile = tl.load(
            b_ptr + n_offs[:, None] * K + k_offs[None, :],
            mask=b_mask, other=0.0
        )   # [BLOCK_N, BLOCK_K], native dtype

        # ── acc[BLOCK_M, BLOCK_N] += A[BLOCK_M, BLOCK_K] @ B.T[BLOCK_K, BLOCK_N] ──
        acc = tl.dot(
            tl.trans(a_tile_T),   # [BLOCK_M, BLOCK_K]
            tl.trans(b_tile),     # [BLOCK_K, BLOCK_N]
            acc,
            out_dtype=tl.float32,
        )

    # ── Store [BLOCK_N, BLOCK_M] transposed ───────────────────────────────
    # c[b, n, h, w] = c_ptr + b * N_size * HW_out + n * HW_out + m_spatial
    # Last dim = m_spatial (stride 1) → COALESCED
    out_mask_T = n_mask[:, None] & m_mask[None, :]
    tl.store(
        c_ptr + b_idx[None, :] * (N_size * HW_out)
              + n_offs[:, None] * HW_out
              + m_spatial[None, :],
        tl.trans(acc).to(a_ptr.dtype.element_ty),
        mask=out_mask_T,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3 (small-M path): fused avg_pool + 1×1 conv in a single pass.
# No intermediate tensor → saves 2 kernel launches + allocation for small N.
# A tile loaded as [BLOCK_K, BLOCK_M] transposed: m_spatial in last dim
# → stride-2 for w, but h0w0+h0w1 together use 100% of each cache line.
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # For M<=1024 (N=1 cases): BLOCK_M=32 → 8-18 tiles; BLOCK_M=16 → 16-36 tiles
        # Larger BLOCK_M (64, 128) gives too few tiles → poor SM occupancy for small M.
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N_size', 'K'],
)
@triton.jit
def _fused_avgpool_conv_small_m(
    inp_ptr,    # NCHW input   [N_batch, C_in,  H_in,  W_in]
    b_ptr,      # weight       [C_out,   C_in]
    c_ptr,      # NCHW output  [N_batch, C_out, H_out, W_out]
    M,          # N_batch * H_out * W_out
    K,          # C_in
    N_size,     # C_out
    HW_out,     # H_out * W_out
    H_in, W_in, W_out,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N

    m_offs = m_start + tl.arange(0, BLOCK_M)
    n_offs = n_start + tl.arange(0, BLOCK_N)

    m_mask = m_offs < M
    n_mask = n_offs < N_size

    b_idx     = m_offs // HW_out
    m_spatial = m_offs % HW_out
    h_out_idx = m_spatial // W_out
    w_out_idx = m_spatial % W_out
    h0 = h_out_idx * 2;  h1 = h0 + 1
    w0 = w_out_idx * 2;  w1 = w0 + 1

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offs < K

        # Load A as [BLOCK_K, BLOCK_M] transposed: m_spatial in last dim.
        # 4 scatter-gather loads (stride-2 in w), averaged → [BLOCK_K, BLOCK_M]
        base  = b_idx * (K * H_in * W_in)          # [BLOCK_M]
        kbase = k_offs * (H_in * W_in)              # [BLOCK_K]
        in_mask_T = k_mask[:, None] & m_mask[None, :]

        a00 = tl.load(inp_ptr + base[None, :] + kbase[:, None] + h0[None, :] * W_in + w0[None, :], mask=in_mask_T, other=0.0).to(tl.float32)
        a10 = tl.load(inp_ptr + base[None, :] + kbase[:, None] + h1[None, :] * W_in + w0[None, :], mask=in_mask_T, other=0.0).to(tl.float32)
        a01 = tl.load(inp_ptr + base[None, :] + kbase[:, None] + h0[None, :] * W_in + w1[None, :], mask=in_mask_T, other=0.0).to(tl.float32)
        a11 = tl.load(inp_ptr + base[None, :] + kbase[:, None] + h1[None, :] * W_in + w1[None, :], mask=in_mask_T, other=0.0).to(tl.float32)

        avg_T = (a00 + a10 + a01 + a11) * 0.25   # [BLOCK_K, BLOCK_M] fp32

        # Load B [BLOCK_N, BLOCK_K]: k in last dim → coalesced
        b_mask = n_mask[:, None] & k_mask[None, :]
        b_tile = tl.load(b_ptr + n_offs[:, None] * K + k_offs[None, :], mask=b_mask, other=0.0)

        # Cast avg_T to native dtype to match b_tile for tl.dot (tensor cores)
        avg_T_native = avg_T.to(b_tile.dtype)

        # acc += avg @ weight.T : [BLOCK_M, BLOCK_K] @ [BLOCK_K, BLOCK_N]
        acc = tl.dot(tl.trans(avg_T_native), tl.trans(b_tile), acc, out_dtype=tl.float32)

    # Store transposed [BLOCK_N, BLOCK_M]: m_spatial in last dim → coalesced
    out_mask_T = n_mask[:, None] & m_mask[None, :]
    tl.store(
        c_ptr + b_idx[None, :] * (N_size * HW_out)
              + n_offs[:, None] * HW_out
              + m_spatial[None, :],
        tl.trans(acc).to(inp_ptr.dtype.element_ty),
        mask=out_mask_T,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Python wrapper (hybrid: single-fused for small M, two-kernel for large M)
# ─────────────────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv_avgpool(in_0, in_1):
    """
    Hybrid dispatch:
    - M <= 1024 (N=1 small batches): single fused kernel (no intermediate tensor)
    - M >  1024 (large batches):     two-kernel (avg_pool → GEMM on pooled tensor)

    in_0: weight [C_out, C_in, 1, 1]
    in_1: input  [N, C_in, H, W]
    → output     [N, C_out, H//2, W//2]
    """
    N_batch, C_in, H_in, W_in = in_1.shape
    C_out  = in_0.shape[0]
    H_out  = H_in // 2
    W_out  = W_in // 2
    HW_out = H_out * W_out
    M      = N_batch * HW_out
    K      = C_in
    N_size = C_out

    weight_2d = in_0.view(C_out, C_in)
    output = torch.empty(
        (N_batch, C_out, H_out, W_out), dtype=in_1.dtype, device=in_1.device
    )

    if M <= 1024:
        # ── Single fused kernel (tiny M: N=1 only) ──────────────────────────
        # Avoids intermediate tensor + 2nd kernel launch overhead.
        grid = lambda meta: (
            triton.cdiv(M,      meta['BLOCK_M']),
            triton.cdiv(N_size, meta['BLOCK_N']),
        )
        _fused_avgpool_conv_small_m[grid](
            in_1, weight_2d, output,
            M, K, N_size, HW_out,
            H_in, W_in, W_out,
        )
    else:
        # ── Two-kernel: avg_pool then GEMM (large M: N=32, 128) ─────────────
        total  = N_batch * C_in * HW_out
        pooled = torch.empty(
            (N_batch, C_in, H_out, W_out), dtype=in_1.dtype, device=in_1.device
        )
        BPOOL = 2048
        _avg_pool2x2_nchw[(triton.cdiv(total, BPOOL),)](
            in_1, pooled,
            H_in, W_in, H_out, W_out, total,
            BLOCK=BPOOL,
        )
        grid = lambda meta: (
            triton.cdiv(M,      meta['BLOCK_M']),
            triton.cdiv(N_size, meta['BLOCK_N']),
        )
        _conv1x1_nchw_gemm[grid](
            pooled, weight_2d, output,
            M, K, N_size, HW_out,
        )

    return output


# ─────────────────────────────────────────────────────────────────────────────
# Replacement entry point
# ─────────────────────────────────────────────────────────────────────────────

def replacement_func():
    return fused_conv_avgpool