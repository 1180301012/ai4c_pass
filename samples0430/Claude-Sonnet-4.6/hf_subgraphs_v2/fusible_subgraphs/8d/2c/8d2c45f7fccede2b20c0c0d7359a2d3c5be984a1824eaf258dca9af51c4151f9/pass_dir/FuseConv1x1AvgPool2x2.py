import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d followed by 2x2 avg_pool2d (stride=2, pad=0,
#          count_include_pad=True).  Fused into a single Triton matmul kernel
#          that averages the 2x2 input windows on-the-fly (pool-first then
#          matmul), eliminating the large intermediate tensor and enabling
#          tensor-core use for float16/bfloat16 inputs.
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    conv2d = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(conv2d, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    # in_0 = weight [OC, IC, 1, 1]
    # in_1 = input  [B,  IC, H,  W]
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel
#   A matrix: "pooled input"  [M, K]   M = B*OH*OW,  K = IC
#             loaded on-the-fly as the avg of 4 spatial neighbours of x
#   B matrix: weight          [K, N]   K = IC, N = OC
#   Output:   out             [B, OC, OH, OW]
#
# Tensor cores are used when input dtype is float16 or bfloat16 by casting
# the averaged tile back to the input dtype before tl.dot.
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # ── large-M configs (big batch, big spatial) ──────────────────────
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        # ── small-M configs (small batch / small spatial — maximise blocks) ─
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def _fused_conv1x1_avgpool2x2_kernel(
    x_ptr,    # [B, IC, H, W]
    w_ptr,    # [OC, IC]  (1x1 weight, treated as 2-D)
    out_ptr,  # [B, OC, OH, OW]
    B, IC, H, W, OC, OH, OW,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # ── swizzled tile scheduling for L2 reuse ──────────────────────────────
    pid        = tl.program_id(0)
    num_pid_m  = tl.cdiv(M, BLOCK_M)
    num_pid_n  = tl.cdiv(N, BLOCK_N)
    num_in_grp = GROUP_M * num_pid_n
    group_id   = pid // num_in_grp
    first_m    = group_id * GROUP_M
    grp_sz_m   = tl.minimum(num_pid_m - first_m, GROUP_M)
    pid_m      = first_m + (pid % num_in_grp) % grp_sz_m
    pid_n      = (pid % num_in_grp) // grp_sz_m

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # row indices [0, M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # col indices [0, N)

    # Decode rm → (b, oh, ow)
    OHOW   = OH * OW
    b_idx  = rm // OHOW
    hw_idx = rm % OHOW
    oh_idx = hw_idx // OW
    ow_idx = hw_idx % OW

    # Precompute pooling spatial offsets broadcast over the K dimension
    h0 = (oh_idx * 2)[:, None]
    h1 = (oh_idx * 2 + 1)[:, None]
    w0 = (ow_idx * 2)[:, None]
    w1 = (ow_idx * 2 + 1)[:, None]
    off_h0w0 = h0 * W + w0
    off_h0w1 = h0 * W + w1
    off_h1w0 = h1 * W + w0
    off_h1w1 = h1 * W + w1

    # Hoist loop-invariant x-base and mask components outside the K loop
    b_x_base = b_idx * (IC * H * W)        # [BLOCK_M]  — constant across ki iters
    m_mask   = (rm < M)[:, None]           # [BLOCK_M, 1]
    rn_K     = rn * K                      # [BLOCK_N]  — for weight address
    n_mask   = rn < N                      # [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for ki in range(0, tl.cdiv(K, BLOCK_K)):
        rk = ki * BLOCK_K + tl.arange(0, BLOCK_K)   # [BLOCK_K]

        # ── load pooled A tile [BLOCK_M, BLOCK_K] ──────────────────────────
        # All loads in native dtype — avoids type-conversion overhead.
        # For fp16/bf16 this matches torch.avg_pool2d's native-dtype behavior.
        k_mask  = (rk < K)[None, :]
        ld_mask = m_mask & k_mask
        x_base  = b_x_base[:, None] + rk[None, :] * (H * W)   # [BLOCK_M, BLOCK_K]

        v00 = tl.load(x_ptr + x_base + off_h0w0, mask=ld_mask, other=0.0)
        v01 = tl.load(x_ptr + x_base + off_h0w1, mask=ld_mask, other=0.0)
        v10 = tl.load(x_ptr + x_base + off_h1w0, mask=ld_mask, other=0.0)
        v11 = tl.load(x_ptr + x_base + off_h1w1, mask=ld_mask, other=0.0)
        a_tile = ((v00 + v01) + (v10 + v11)) * 0.25   # native-dtype average

        # ── load weight B tile [BLOCK_K, BLOCK_N] ──────────────────────────
        w_offs = rn_K[None, :] + rk[:, None]          # [BLOCK_K, BLOCK_N]
        w_mask = (rk[:, None] < K) & n_mask[None, :]
        b_tile = tl.load(w_ptr + w_offs, mask=w_mask, other=0.0)

        # ── matmul with fp32 accumulation ───────────────────────────────────
        # For fp16/bf16: uses tensor cores (HMMA) automatically.
        # For fp32: uses tf32 tensor cores (Triton default for fp32 dot).
        acc += tl.dot(a_tile, b_tile, out_dtype=tl.float32)

    # ── store output [B, OC, OH, OW] ───────────────────────────────────────
    b_o  = rm // OHOW
    hw_o = rm % OHOW
    oh_o = hw_o // OW
    ow_o = hw_o % OW

    out_off = (b_o[:, None]  * (OC * OH * OW) +
               rn[None, :]   * (OH * OW) +
               oh_o[:, None] * OW +
               ow_o[:, None])

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr + out_off, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


# ---------------------------------------------------------------------------
# Wrapper — returns a plain tensor (not a tuple) to match pattern output
# ---------------------------------------------------------------------------

@torch.fx.wrap
def _fused_conv1x1_avgpool(in_0, in_1):
    """
    in_0 : weight  [OC, IC, 1, 1]
    in_1 : input   [B,  IC, H,  W]
    Returns a plain tensor of shape [B, OC, H//2, W//2].
    """
    x = in_1   # [B, IC, H, W]
    w = in_0   # [OC, IC, 1, 1] — flat-indexed as [OC, IC]

    B   = x.shape[0]
    IC  = x.shape[1]
    H   = x.shape[2]
    W   = x.shape[3]
    OC  = w.shape[0]
    OH  = H // 2
    OW  = W // 2
    M   = B * OH * OW
    N   = OC
    K   = IC

    out = torch.empty((B, OC, OH, OW), dtype=x.dtype, device=x.device)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    _fused_conv1x1_avgpool2x2_kernel[grid](
        x, w, out,
        B, IC, H, W, OC, OH, OW,
        M, N, K,
    )

    return out


def replacement_func():
    return _fused_conv1x1_avgpool