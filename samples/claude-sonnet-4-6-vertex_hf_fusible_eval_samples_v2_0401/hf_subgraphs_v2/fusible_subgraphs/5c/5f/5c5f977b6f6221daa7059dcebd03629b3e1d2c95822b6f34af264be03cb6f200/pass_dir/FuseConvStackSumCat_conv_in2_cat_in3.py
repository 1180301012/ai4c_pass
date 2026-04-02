"""
Pattern A: conv2d(in_2, in_1, in_0, ...) -> stack([x], 0) -> sum(0) -> cat([x, in_3], 1)

Key observations:
  1. stack([x], dim=0).sum(dim=0) is a no-op identity — eliminates a huge temp allocation
  2. 1x1 conv = batched GEMM: for each n, out[n] = weight @ input[n] + bias
  3. Cat can be eliminated by writing GEMM directly into the output buffer at offset 0,
     then copying cat_input into the output buffer at offset OC.
     This removes the intermediate conv_out allocation AND halves the cat bandwidth.

Memory bandwidth (old vs new), with OC≈OC2:
  Old: N*(IC + 5*OC)*HW*size  (GEMM-write + cat-read-A + cat-read-B + cat-write)
  New: N*(IC + 3*OC)*HW*size  (GEMM writes straight to output + one copy for B)
"""
import torch
import triton
import triton.language as tl


# ─── Kernel 1: 1×1 conv as batched GEMM — writes directly into output buffer ─
# 3D grid: (cdiv(OC, BLOCK_M), cdiv(HW, BLOCK_N), N)  — avoids div/mod in kernel

@triton.autotune(
    configs=[
        # ── Smallest tiles: best SM occupancy, good for N=1 ──────────────────
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        # ── Small tiles ──────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 512, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        # ── Medium tiles ─────────────────────────────────────────────────────
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        # ── BLOCK_K=128: only 2 K-loop iters for IC=256 ──────────────────────
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
    ],
    key=['N', 'IC', 'OC', 'HW'],
)
@triton.jit
def conv1x1_gemm_into_a(
    w_ptr,     # weight [OC, IC]  (1×1 kernel squeezed)
    b_ptr,     # bias   [OC]
    x_ptr,     # input  [N, IC, HW]
    out_ptr,   # output [N, TOTAL_C, HW]  — writes to channels [OC_OFFSET, OC_OFFSET+OC)
    N, IC, OC, HW,
    TOTAL_C,    # OC + OC2 — stride for N dimension in out
    OC_OFFSET,  # first output channel to write (= 0 for the conv part)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Grid: (cdiv(OC, BLOCK_M), cdiv(HW, BLOCK_N), N)
    # OC tile (pid_m) varies FASTEST: programs with same (pid_ni, pid_n) share the
    # same input HW tile [IC, BLOCK_N] in L1/L2 across all OC groups → input reuse.
    # The input tensor (N×IC×HW) is large, so reusing it is more valuable than
    # reusing the small weight tensor (OC×IC).
    pid_m  = tl.program_id(0)   # OC tile index   (fastest → input HW tile reused across OC)
    pid_ni = tl.program_id(1)   # HW tile index
    pid_n  = tl.program_id(2)   # batch index     (slowest)

    m_offs = pid_m  * BLOCK_M + tl.arange(0, BLOCK_M)   # local OC indices
    n_offs = pid_ni * BLOCK_N + tl.arange(0, BLOCK_N)   # HW indices

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_step in range(0, tl.cdiv(IC, BLOCK_K)):
        k_offs = k_step * BLOCK_K + tl.arange(0, BLOCK_K)

        # weight[oc, ic]
        w = tl.load(w_ptr + m_offs[:, None] * IC + k_offs[None, :],
                    mask=(m_offs[:, None] < OC) & (k_offs[None, :] < IC),
                    other=0.0)

        # input[n, ic, hw]
        x = tl.load(x_ptr + pid_n * IC * HW + k_offs[:, None] * HW + n_offs[None, :],
                    mask=(k_offs[:, None] < IC) & (n_offs[None, :] < HW),
                    other=0.0)

        acc = tl.dot(w, x, acc)

    # Bias
    b = tl.load(b_ptr + m_offs, mask=m_offs < OC, other=0.0)
    acc += b[:, None]

    # Write to out[n, OC_OFFSET+oc, hw] — stride is TOTAL_C in channel dimension
    out_c = OC_OFFSET + m_offs
    tl.store(
        out_ptr + pid_n * TOTAL_C * HW + out_c[:, None] * HW + n_offs[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=(m_offs[:, None] < OC) & (n_offs[None, :] < HW),
    )


# ─── Kernel 2: copy src[N, C_SRC, HW] → dst[N, DST_TOTAL_C, HW] at offset ──

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256},  num_warps=4),
        triton.Config({'BLOCK_HW': 512},  num_warps=4),
        triton.Config({'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_HW': 2048}, num_warps=8),
        triton.Config({'BLOCK_HW': 4096}, num_warps=8),
    ],
    key=['N', 'C_SRC', 'HW'],
)
@triton.jit
def copy_into_a(
    src_ptr, dst_ptr,
    N, C_SRC, HW,
    DST_TOTAL_C,   # total channels in dst
    DST_OFFSET,    # where to start writing (= OC)
    BLOCK_HW: tl.constexpr,
):
    # Grid: (N * C_SRC,  cdiv(HW, BLOCK_HW))
    pid_nc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    n = pid_nc // C_SRC
    c = pid_nc % C_SRC

    hw_start = pid_hw * BLOCK_HW
    hw_offs  = hw_start + tl.arange(0, BLOCK_HW)
    mask     = hw_offs < HW

    data = tl.load(src_ptr + (n * C_SRC + c) * HW + hw_offs, mask=mask, other=0.0)

    dst_c = DST_OFFSET + c
    tl.store(dst_ptr + (n * DST_TOTAL_C + dst_c) * HW + hw_offs, data, mask=mask)


# ─── Wrapper ─────────────────────────────────────────────────────────────────

@torch.fx.wrap
def fused_conv_stack_sum_cat_a(bias, weight, conv_input, cat_input):
    """
    Replaces: conv2d(in_2,...) → stack([x],0) → sum(0) → cat([x, cat_input], 1)

    Optimization:
    - Allocate final output[N, OC+OC2, H, W] once
    - Kernel 1 (GEMM, 3D grid): writes output[:, :OC, :]   (no intermediate conv_out!)
    - Kernel 2 (copy): writes output[:, OC:, :]   (single-source read of cat_input)
    """
    N, IC, H, W = conv_input.shape
    OC      = weight.shape[0]
    OC2     = cat_input.shape[1]
    HW      = H * W
    TOTAL_C = OC + OC2

    output = torch.empty((N, TOTAL_C, H, W),
                         dtype=conv_input.dtype, device=conv_input.device)

    # --- GEMM → first OC channels (3D grid: OC fast, HW mid, N slow) ---
    grid_gemm = lambda meta: (
        triton.cdiv(OC, meta['BLOCK_M']),  # OC tiles (fastest → input HW tile reused)
        triton.cdiv(HW, meta['BLOCK_N']),  # HW tiles
        N,                                  # batch (slowest)
    )
    conv1x1_gemm_into_a[grid_gemm](
        weight, bias, conv_input, output,
        N, IC, OC, HW,
        TOTAL_C, 0,   # writes to channel [0, OC)
    )

    # --- Copy → last OC2 channels ---
    grid_copy = lambda meta: (
        N * OC2,
        triton.cdiv(HW, meta['BLOCK_HW']),
    )
    copy_into_a[grid_copy](
        cat_input, output,
        N, OC2, HW,
        TOTAL_C, OC,  # writes to channel [OC, OC+OC2)
    )

    return output


# ─── Pattern / Replacement API ───────────────────────────────────────────────

def pattern(in_0, in_1, in_2, in_3):
    """Matches: conv2d(in_2, in_1, in_0, ...) → stack → sum → cat with in_3"""
    conv2d = torch.conv2d(in_2, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    tmp_3  = torch.stack([conv2d], dim=0)
    tmp_4  = tmp_3.sum(dim=0)
    tmp_5  = torch.cat([tmp_4, in_3], 1)
    return tmp_5


def replacement_args(in_0, in_1, in_2, in_3):
    # (bias, weight, conv_input, cat_input)
    return (in_0, in_1, in_2, in_3)


def replacement_func():
    return fused_conv_stack_sum_cat_a