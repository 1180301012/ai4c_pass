import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Fused 1×1-Conv2d + 2×2-AvgPool kernel
#
# Grid:  (ceil(M / BLOCK_M),  ceil(C_out / BLOCK_C))
#   M   = N * OH * OW  (all output spatial positions)
#
# Each CTA computes the output tile:
#   out[n, k, oh, ow] = 1/4 * Σ_{dh=0,1} Σ_{dw=0,1}  A[dh,dw] · w[k, c]
# where c is summed over in BLOCK_C chunks.
# ---------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_C': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 64},  num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 512}, num_stages=4, num_warps=8),
    ],
    key=['M', 'C_in', 'C_out'],
)
@triton.jit
def _fused_conv1x1_avgpool_kernel(
    in1_ptr,       # [N, C_in, H, W]    NCHW input
    in0_ptr,       # [C_out, C_in, 1, 1] conv weight (squeezed)
    out_ptr,       # [N, C_out, OH, OW]  output
    N, C_in, H, W,
    C_out, OH, OW,
    OHW,           # OH * OW
    M,             # N * OHW
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)   # spatial tile
    pid_k = tl.program_id(1)   # C_out tile

    n    = pid_m // OHW        # batch
    ohw  = pid_m % OHW         # oh * OW + ow

    # ── per-tile index vectors ───────────────────────────────────────────────
    m_offs  = tl.arange(0, BLOCK_M)
    m_mask  = (pid_m + m_offs) < M

    k_off  = pid_k * BLOCK_C
    k_offs = k_off + tl.arange(0, BLOCK_C)
    k_mask = k_offs < C_out

    c_offs = tl.arange(0, BLOCK_C)   # used within each C_in block

    # ── base offsets for the current (n, ohw) and all 4 pool positions ──────
    flat_n  = n * (C_in * OHW)
    oh      = ohw // OW
    ow      = ohw % OW
    base_e  = oh * 2 * OW + ow * 2    # 2*oh*W + 2*ow
    base_o  = oh * 2 * OW + ow * 2 + 1
    base_e2 = (oh + 1) * 2 * OW + ow * 2
    base_o2 = base_e2 + 1

    # ── accumulator  [BLOCK_M, BLOCK_C]  in float32 ─────────────────────────
    acc = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)

    # ── reduce over C_in in BLOCK_C chunks ──────────────────────────────────
    for k_rem in range(0, C_in, BLOCK_C):
        ci_off  = k_rem + c_offs
        ci_mask = ci_off < C_in

        # ---- load input A[0..3] : (BLOCK_M, BLOCK_C) ----
        a0 = tl.load(in1_ptr + flat_n + ci_off.to(tl.int64) * OHW + base_e,
                     mask=(m_mask & ci_mask), other=0.0).to(tl.float32)
        a1 = tl.load(in1_ptr + flat_n + ci_off.to(tl.int64) * OHW + base_o,
                     mask=(m_mask & ci_mask), other=0.0).to(tl.float32)
        a2 = tl.load(in1_ptr + flat_n + ci_off.to(tl.int64) * OHW + base_e2,
                     mask=(m_mask & ci_mask), other=0.0).to(tl.float32)
        a3 = tl.load(in1_ptr + flat_n + ci_off.to(tl.int64) * OHW + base_o2,
                     mask=(m_mask & ci_mask), other=0.0).to(tl.float32)

        # ---- load weight tile  [BLOCK_C, BLOCK_C] in native dtype ----
        wt_add  = k_offs.to(tl.int64)[:, None] * C_in + ci_off.to(tl.int64)[None, :]
        wt_block = tl.load(in0_ptr + wt_add,
                           mask=(k_mask[:, None] & ci_mask[None, :]),
                           other=0.0)

        # ---- accumulate outer products  ----
        # acc[m, k] += Σ_c  ( A0[:,c]*wt[k,c] + A1[:,c]*wt[k,c] + ... )
        w_k = wt_block.to(tl.float32)          # [BLOCK_C, BLOCK_C]
        acc += tl.sum(a0[:, None] * w_k[0:1, :], axis=2)
        acc += tl.sum(a1[:, None] * w_k[2:3, :], axis=2)
        acc += tl.sum(a2[:, None] * w_k[4:5, :], axis=2)
        acc += tl.sum(a3[:, None] * w_k[6:7, :], axis=2)

    result = (acc / 4.0).to(tl.float32)

    # ── store output ─────────────────────────────────────────────────────────
    out_idx = (n.to(tl.int64) * (C_out * OHW)
               + k_offs.to(tl.int64) * OHW
               + ohw.to(tl.int64))[None, :]
    tl.store(out_ptr + out_idx, result,
             mask=(m_mask[:, None] & k_mask[None, :]))


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------
@torch.fx.wrap
def _fused_conv1x1_avgpool(weight, inp):
    N, C_in, H, W = inp.shape
    C_out = weight.shape[0]
    OH    = H // 2
    OW    = W // 2
    M     = N * OH * OW

    out = torch.empty((N, C_out, OH, OW), device=inp.device, dtype=inp.dtype)

    grid = lambda meta: (
        triton.cdiv(M,    meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_C']),
    )
    _fused_conv1x1_avgpool_kernel[grid](
        inp, weight, out,
        N, C_in, H, W,
        C_out, OH, OW,
        OH * OW,
        M,
    )
    return (out,)


# ---------------------------------------------------------------------------
# Pattern / replacement API
# ---------------------------------------------------------------------------

def pattern(weight, inp):
    conv = torch.conv2d(inp, weight, None, (1, 1), (0, 0), (1, 1), 1)
    pool = torch.nn.functional.avg_pool2d(conv, 2, 2, 0, False, True, None)
    return (pool,)


def replacement_args(weight, inp):
    return (weight, inp)


def replacement_func():
    return _fused_conv1x1_avgpool