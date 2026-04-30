import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: 1x1 conv2d  ->  avg_pool2d(kernel=2, stride=2, pad=0, count_include_pad=True)
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_1 = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    tmp_2 = torch.nn.functional.avg_pool2d(tmp_1, 2, 2, 0, False, True, None)
    return tmp_2


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Triton kernel: fused 1x1-conv + 2x2 avg-pool
#
# Key idea: for each 4-corner 2x2 tile around output position (oh, ow),
# compute the conv sum for x[n, c_in, 2*oh, 2*ow+kx] and weight sums, then
# average the four sums.
#
# Grid: (N, ceil(OHW_out / BLOCK_OHW), ceil(C_out / BLOCK_COUT))
#
# x layout : [N, C_in, H, W]   stride: (C_in*H*W, H*W, W, 1)
# w layout : [C_out, C_in]     stride: (C_in, 1)
# out layout: [N, C_out, OHW_out] (contiguous H*W plane)
# ---------------------------------------------------------------------------

@triton.autotune(
    configs=[
        # (BLOCK_OHW, BLOCK_CIN, BLOCK_COUT)  -- keep BLOCK_OHW*BLOCK_COUT < ~32K to avoid register spill
        triton.Config({'BLOCK_OHW': 16, 'BLOCK_CIN': 32, 'BLOCK_COUT': 64},  num_warps=4),
        triton.Config({'BLOCK_OHW': 16, 'BLOCK_CIN': 32, 'BLOCK_COUT': 128}, num_warps=4),
        triton.Config({'BLOCK_OHW': 16, 'BLOCK_CIN': 64, 'BLOCK_COUT': 64},  num_warps=4),
        triton.Config({'BLOCK_OHW': 32, 'BLOCK_CIN': 32, 'BLOCK_COUT': 64},  num_warps=4),
        triton.Config({'BLOCK_OHW': 32, 'BLOCK_CIN': 32, 'BLOCK_COUT': 128}, num_warps=4),
        triton.Config({'BLOCK_OHW': 32, 'BLOCK_CIN': 64, 'BLOCK_COUT': 64},  num_warps=4),
        triton.Config({'BLOCK_OHW': 64, 'BLOCK_CIN': 16, 'BLOCK_COUT': 64},  num_warps=4),
        triton.Config({'BLOCK_OHW': 64, 'BLOCK_CIN': 32, 'BLOCK_COUT': 64},  num_warps=4),
        triton.Config({'BLOCK_OHW': 64, 'BLOCK_CIN': 32, 'BLOCK_COUT': 128}, num_warps=4),
        triton.Config({'BLOCK_OHW': 64, 'BLOCK_CIN': 64, 'BLOCK_COUT': 64},  num_warps=8),
    ],
    key=['N', 'C_in', 'C_out', 'OHW_out'],
)
@triton.jit
def _fused_conv1x1_avgpool_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, C_out, H, W, OHW_out, OW_out,
    BLOCK_OHW:  tl.constexpr,
    BLOCK_CIN:  tl.constexpr,
    BLOCK_COUT: tl.constexpr,
):
    # ---- program indices ----
    n_idx       = tl.program_id(0)
    ohw_tile    = tl.program_id(1)
    cout_tile   = tl.program_id(2)

    ohw_start   = ohw_tile * BLOCK_OHW
    c_out_start = cout_tile * BLOCK_COUT

    # ---- index ranges ----
    ohw_offs   = ohw_start   + tl.arange(0, BLOCK_OHW)    # [BLOCK_OHW]
    c_out_offs = c_out_start + tl.arange(0, BLOCK_COUT)   # [BLOCK_COUT]

    ohw_mask   = ohw_offs   < OHW_out
    c_out_mask = c_out_offs < C_out

    oh = ohw_offs // OW_out   # [BLOCK_OHW]
    ow = ohw_offs % OW_out    # [BLOCK_OHW]

    # 2x2 corner base addresses  [BLOCK_OHW]
    ih0 = 2 * oh
    iw0 = 2 * ow

    # ---- accumulators for the 4 pooling corners ----
    # shape [BLOCK_OHW, BLOCK_COUT] in fp32
    acc00 = tl.zeros([BLOCK_OHW, BLOCK_COUT], dtype=tl.float32)
    acc01 = tl.zeros([BLOCK_OHW, BLOCK_COUT], dtype=tl.float32)
    acc10 = tl.zeros([BLOCK_OHW, BLOCK_COUT], dtype=tl.float32)
    acc11 = tl.zeros([BLOCK_OHW, BLOCK_COUT], dtype=tl.float32)

    x_base = n_idx * C_in * H * W

    # ---- loop over C_in in BLOCK_CIN steps ----
    for c_in_start in tl.range(0, C_in, BLOCK_CIN):
        c_in_offs = c_in_start + tl.arange(0, BLOCK_CIN)   # [BLOCK_CIN]
        c_in_mask = c_in_offs < C_in

        # Load weight slice [BLOCK_COUT, BLOCK_CIN]  (contiguous along C_in)
        # in_0 is [C_out, C_in, 1, 1] → element [c_out, c_in] = w_ptr + c_out*C_in + c_in
        w_ptrs = w_ptr + c_out_offs[:, None] * C_in + c_in_offs[None, :]
        wn = tl.load(w_ptrs,
                     mask=c_out_mask[:, None] & c_in_mask[None, :],
                     other=0.0).to(tl.float32)   # [BLOCK_COUT, BLOCK_CIN]

        # Load the 4 pooling corners for this C_in chunk
        # xc_kk[i, j] = flat index for x at (n, c_in_offs[j], oh*2+dh, ow*2+dw)
        xc00 = (x_base
                + c_in_offs[None, :] * (H * W)
                + ih0[:, None] * W + iw0[:, None])        # [BLOCK_OHW, BLOCK_CIN]
        xc01 = xc00 + 1
        xc10 = xc00 + W
        xc11 = xc00 + W + 1

        xm00 = tl.load(x_ptr + xc00,
                       mask=ohw_mask[:, None] & c_in_mask[None, :],
                       other=0.0).to(tl.float32)
        xm01 = tl.load(x_ptr + xc01,
                       mask=ohw_mask[:, None] & c_in_mask[None, :],
                       other=0.0).to(tl.float32)
        xm10 = tl.load(x_ptr + xc10,
                       mask=ohw_mask[:, None] & c_in_mask[None, :],
                       other=0.0).to(tl.float32)
        xm11 = tl.load(x_ptr + xc11,
                       mask=ohw_mask[:, None] & c_in_mask[None, :],
                       other=0.0).to(tl.float32)

        # GEMM: [BLOCK_OHW, BLOCK_COUT] += [BLOCK_OHW, BLOCK_CIN] @ [BLOCK_CIN, BLOCK_COUT]
        acc00 = tl.dot(xm00, tl.trans(wn), acc00, allow_tf32=False)
        acc01 = tl.dot(xm01, tl.trans(wn), acc01, allow_tf32=False)
        acc10 = tl.dot(xm10, tl.trans(wn), acc10, allow_tf32=False)
        acc11 = tl.dot(xm11, tl.trans(wn), acc11, allow_tf32=False)

    # ---- avg-pool: average the 4 corners ----
    out_val = (acc00 + acc01 + acc10 + acc11) * 0.25   # [BLOCK_OHW, BLOCK_COUT]

    # ---- store to output[n, c_out, ohw] ----
    out_ptrs = (out_ptr
                + n_idx      * C_out * OHW_out
                + c_out_offs[None, :] * OHW_out
                + ohw_offs[:, None])

    tl.store(out_ptrs,
             out_val.to(out_ptr.dtype.element_ty),
             mask=ohw_mask[:, None] & c_out_mask[None, :])


# ---------------------------------------------------------------------------
# Kernel wrapper
# ---------------------------------------------------------------------------

@torch.fx.wrap
def fused_conv1x1_avgpool(in_0, in_1):
    """
    in_0 : weight  [C_out, C_in, 1, 1]
    in_1 : input   [N,    C_in, H, W]

    Returns [N, C_out, OH_out, OW_out]
      OH_out = (H - 2) // 2 + 1,  OW_out = (W - 2) // 2 + 1
    """
    N, C_in, H, W = in_1.shape
    C_out = in_0.shape[0]

    # avg_pool2d(kernel=2, stride=2, pad=0, count_include_pad=True)
    OH_out = (H - 2) // 2 + 1
    OW_out = (W - 2) // 2 + 1
    OHW_out = OH_out * OW_out

    out = torch.empty((N, C_out, OH_out, OW_out),
                      dtype=in_1.dtype, device=in_1.device)

    grid = lambda meta: (
        N,
        triton.cdiv(OHW_out, meta['BLOCK_OHW']),
        triton.cdiv(C_out,    meta['BLOCK_COUT']),
    )

    _fused_conv1x1_avgpool_kernel[grid](
        in_1, in_0, out,
        N, C_in, C_out, H, W, OHW_out, OW_out,
    )

    return out


# ---------------------------------------------------------------------------
# Required by the AI4C framework
# ---------------------------------------------------------------------------

def replacement_func():
    return fused_conv1x1_avgpool