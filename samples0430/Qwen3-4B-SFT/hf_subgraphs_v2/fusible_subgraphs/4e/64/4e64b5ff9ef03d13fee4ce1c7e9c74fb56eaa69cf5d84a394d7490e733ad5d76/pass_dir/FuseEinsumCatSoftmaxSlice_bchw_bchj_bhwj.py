import torch
import triton
import triton.language as tl


# ──────────────────────────────── pattern ────────────────────────────────

def pattern(in_0, in_1, in_2):
    einsum = torch.functional.einsum('bchw,bchj->bhwj', in_2, in_1)
    tmp_2 = torch.cat([in_0, einsum], dim=-1)
    tmp_3 = torch.nn.functional.softmax(tmp_2, dim=-1)
    tmp_4 = tmp_3[(Ellipsis, slice(None, 64, None))]
    return (tmp_3, tmp_4)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


# ─────────────────────────── kernel definitions ────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_D': 16, 'BLOCK_J': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 32, 'BLOCK_J': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_D': 64, 'BLOCK_J': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_D': 16, 'BLOCK_J': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_D': 32, 'BLOCK_J': 64}, num_warps=8, num_stages=3),
    ],
    key=['B', 'C_h', 'H', 'W'],
)
@triton.jit
def _fused_einsum_cat_softmax_kernel(
    # input pointers
    in0_ptr,      # in_0  [B, C_h, H, W]  ── first-half of cat
    in1_ptr,      # in_1  [B, C_h, W, J]  ── key  (for einsum)
    in2_ptr,      # in_2  [B, C_h, H, W, D=64] ── query (for einsum)
    # output pointers
    out0_ptr,     # tmp_4  [B, H, W, J]   ── first J=64 cols (in_0 part)
    out1_ptr,     # tmp_3  [B, H, W, 2J]  ── full softmax output
    # shape
    B, C_h, H, W,
    # meta
    IS_BF16: tl.constexpr,
    J:       tl.constexpr,
    C:       tl.constexpr,   # = C_h (≡ D in einsum notation)
    # tile sizes
    BLOCK_D: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    """
    One program per (b, h, w) row.
    Fuses:
      - einsum('bchw,bchj->bhwj')              (D=64 dot products over c_h)
      - cat([in_0, einsum], dim=-1)             (shift einsum to j in [J,2J))
      - softmax( concat, dim=-1 )               (online softmax over 128 elems)
    """
    # ── decode program id → (b, h, w) ──────────────────────────────────
    HW        = H * W
    row_count = B * HW
    prog_id   = tl.program_id(0)
    hw        = prog_id % HW
    b         = prog_id // HW
    h_idx     = hw // W
    w_idx     = hw % W

    # local channel indices for this row
    c_offs = tl.arange(0, C)          # [0 .. C-1]
    j_offs = tl.arange(0, BLOCK_J)    # [0 .. J-1]

    # base offsets (flat indices into each tensor's memory)
    in0_off  = b * (C * H * W) + h_idx * (W) + w_idx   # in_0[b, h, w, 0]
    in1_base = b * (C * W * J) + w_idx * J              # in_1[b, ?, w, 0]
    in2_base = b * (C * H * W * C) + h_idx * (W * C) + w_idx * C

    # ── soft-mask utilities ─────────────────────────────────────────────
    mask_j   = j_offs < J                               # always full for J=BLOCK_J=64
    mask_a   = (j_offs >= J) & mask_j
    mask_sc  = mask_j & mask_a
    mask_tot = mask_j                            # valid positions across all (a, s) halves

    # ── affine-transform of patch index ─────────────────────────────────
    a = j_offs // J   # patch index: 0 for in_0 half, 1..C for einsum half

    # ── einsum accumulation ─────────────────────────────────────────────
    # We want acc[c_h] = sum_{d=0}^{C-1} in2[b,c_h,h,w,d] * in1[b,c_h,w,j]
    # Iterate in BLOCK_D steps to avoid extreme register pressure.
    # After the D-loop the accumulator y_acc is [1, C] where y_acc[0,c_h] = acc[c_h].
    y_acc = tl.full([1, C], 0.0, dtype=tl.float32)

    for d_start in range(0, C, BLOCK_D):
        d_offs = d_start + tl.arange(0, BLOCK_D)   # [BLOCK_D]

        # query slice  in_2[b, c_h, h, w, d]  → shape [ BLOCK_D ]
        q = tl.load(
            in2_ptr + in2_base + d_offs[:, None] * C + c_offs[None, :],
            mask=(d_offs[:, None] < C - d_start) & (c_offs[None, :] < C),
            other=0.0,
        )

        # key slice  in_1[b, c_h, w, j]  → shape [ BLOCK_D, BLOCK_J ]
        k = tl.load(
            in1_ptr + in1_base + d_offs[:, None] * J + j_offs[None, :],
            mask=(d_offs[:, None] < C - d_start) & (j_offs[None, :] < J),
            other=0.0,
        )                    # [BLOCK_D, BLOCK_J]

        # acc[c] += dot( q, k, over dim-0 )
        # q:[BLOCK_D],  k:[BLOCK_D, BLOCK_J]  →  q[:,None]*k -> [BLOCK_D,BLOCK_J] -> sum 0-axis -> [BLOCK_J]
        # We broadcast q to [BLOCK_D, 1] and sum axis-0:
        #   q[: , None] * k   = [BLOCK_D, BLOCK_J]
        #   tl.sum(…, axis=0) = [BLOCK_J]   (each element acc[j]
        #                             += Σ_d  q[d] * k[d, j])
        q_2d   = tl.expand_dims(q, 1)           # [BLOCK_D, 1]
        qk     = q_2d * k                       # [BLOCK_D, BLOCK_J]
        qk_sum = tl.sum(qk, axis=0)             # [BLOCK_J]

        # add to acc: scalar-wise broadcast from [BLOCK_J] to y_acc[0, *]
        y_acc = y_acc + tl.expand_dims(qk_sum.to(tl.float32), 0)

    # ── extract scalar y[0] = y_acc[0, 0] ───────────────────────────────
    y_small = tl.sum(y_acc, axis=1)             # [1]
    y0      = y_small[0, 0]                     # scalar

    # ── assemble 128-element row for softmax ────────────────────────────
    x = tl.zeros([128], dtype=tl.float32)

    # first J elements ← from in_0  (already added in-place)
    x[0:J] = tl.load(
        in0_ptr + in0_off + j_offs,
        mask=mask_j, other=-1e9,
    ).to(tl.float32)

    # last J elements ← einsum result
    x[J:2 * J] = tl.where(
        mask_a,
        y0 + tl.expand_dims(j_offs - J, 1).to(tl.float32),   # [1, BLOCK_J]
        -1e9,
    )

    # ── online softmax over the 128-element row ─────────────────────────
    x_max   = tl.max(x, axis=0)
    x_minf  = tl.where(mask_tot, x - x_max, -1e10)
    x_exp   = tl.where(mask_tot, tl.exp(x_minf), 0.0)
    x_sum   = tl.sum(x_exp, axis=0)
    y_exp   = x_exp / x_sum                        # [128]

    # ── store tmp_4 (first half = in_0 part) ────────────────────────────
    out1_base = b * (H * W * 2 * J) + h_idx * (W * 2 * J) + w_idx * (2 * J)
    tl.store(out0_ptr + out1_base + j_offs,
             y_exp[0:J].to(tl.float32),
             mask=mask_j)

    # ── store tmp_3 (full softmax, first 64 cols = out0_base+j; second 64 = out1_base+J+j) ─
    slices_ptr = out1_ptr + out1_base + J          # same flat offset as out0_ptr + out1_base
    tl.store(out0_ptr + out1_base + j_offs + J,
             y_exp[J:2 * J].to(tl.float32),
             mask=mask_a)
    tl.store(slices_ptr + j_offs,
             y_exp[J:2 * J].to(tl.float32),
             mask=mask_a)


# ─────────────────── wrapper (called by the replacement) ─────────────────

@torch.fx.wrap
def fused_einsum_cat_softmax(in_0, in_1, in_2):
    """
    in_0 : [B, C_h, H, W]           (first-half of cat)
    in_1 : [B, C_h, W, J]           (key  for einsum)
    in_2 : [B, C_h, H, W, D=64]     (query for einsum)

    Returns (tmp_3, tmp_4) where:
        tmp_3  : [B, H, W, 2J]  full  softmax output
        tmp_4  : [B, H, W, J]   tmp_3[..., :J]
    """
    B, C_h, H, W = in_0.shape
    J = in_1.shape[3]   # = 64

    IS_BF16 = (in_0.dtype == torch.bfloat16)

    tmp_3 = torch.empty((B, H, W, 2 * J), dtype=in_0.dtype, device=in_0.device)
    tmp_4 = torch.empty((B, H, W, 2 * J), dtype=in_0.dtype, device=in_0.device)

    total_rows = B * H * W
    grid = (total_rows,)

    _fused_einsum_cat_softmax_kernel[grid](
        in_0, in_1, in_2,
        tmp_4, tmp_3,
        B, C_h, H, W,
        IS_BF16=IS_BF16,
        J=J,
        C=C_h,
    )

    return (tmp_3, tmp_4)


# ─────────────────-------- replacement entry ─────────────────────────────

def replacement_func():
    return fused_einsum_cat_softmax