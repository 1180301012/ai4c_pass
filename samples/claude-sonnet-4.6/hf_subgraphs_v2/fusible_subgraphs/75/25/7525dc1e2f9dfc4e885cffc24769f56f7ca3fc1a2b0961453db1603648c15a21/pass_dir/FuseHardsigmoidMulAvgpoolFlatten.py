import torch
import triton
import triton.language as tl


def pattern(conv2d_out, in2):
    hs = torch.nn.functional.hardsigmoid(conv2d_out, False)
    mul = in2 * hs
    pool = torch.nn.functional.adaptive_avg_pool2d(mul, 1)
    flat = pool.flatten(1, -1)
    drop = torch.nn.functional.dropout(flat, 0.0, False, False)
    return drop


def replacement_args(conv2d_out, in2):
    return (conv2d_out, in2)


# ─────────────────────────────────────────────────────────────────────────────
# 2-D tiled kernel:  each CTA handles BLOCK_C consecutive channels for one
# batch item, and accumulates their mean over H*W in a single pass.
#
# out[b, c] = hardsigmoid(conv_out[b,c,0,0]) * mean(in2[b,c,:,:])
#
# Grid: (B * ceil(C / BLOCK_C),)
# ─────────────────────────────────────────────────────────────────────────────
@triton.jit
def fused_hs_mul_avgpool_kernel(
    conv_out_ptr,               # [B, C, 1, 1]  contiguous
    in2_ptr,                    # [B, C, H, W]  contiguous
    out_ptr,                    # [B, C]        output
    B, C, HW,
    BLOCK_C:  tl.constexpr,    # channels per CTA
    BLOCK_HW: tl.constexpr,    # HW tile size
):
    pid        = tl.program_id(0)
    n_c_tiles  = tl.cdiv(C, BLOCK_C)
    pid_b      = pid // n_c_tiles
    pid_c_tile = pid % n_c_tiles
    c_start    = pid_c_tile * BLOCK_C

    # ---- channel offsets [BLOCK_C] ----------------------------------------
    c_offs = c_start + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C                                # handle C % BLOCK_C != 0

    # ---- hardsigmoid(conv_out[b, c, 0, 0]) --------------------------------
    # conv_out [B, C, 1, 1] contiguous → offset = b*C + c
    conv_vals = tl.load(conv_out_ptr + pid_b * C + c_offs,
                        mask=c_mask, other=0.0).to(tl.float32)
    hs = tl.minimum(tl.maximum(conv_vals * 0.16666667 + 0.5, 0.0), 1.0)

    # ---- mean(in2[b, c, :, :]) for BLOCK_C channels concurrently ----------
    sums    = tl.zeros([BLOCK_C], dtype=tl.float32)
    hw_base = pid_b * C * HW

    for i in range(0, tl.cdiv(HW, BLOCK_HW)):
        hw_offs = i * BLOCK_HW + tl.arange(0, BLOCK_HW)   # [BLOCK_HW]

        # Build an explicit [BLOCK_C, BLOCK_HW] index and mask.
        # mask_2d[dc, dh] = (c valid) AND (hw offset < HW)
        # Using outer-product of the two 1-D masks avoids implicit broadcasting,
        # which is the root cause of crashes for some (dtype, shape) combos.
        idx     = hw_base + c_offs[:, None] * HW + hw_offs[None, :]   # [BLOCK_C, BLOCK_HW]
        mask_2d = c_mask[:, None] & (hw_offs[None, :] < HW)           # [BLOCK_C, BLOCK_HW]

        vals  = tl.load(in2_ptr + idx, mask=mask_2d, other=0.0).to(tl.float32)
        sums += tl.sum(vals, axis=1)       # reduce over BLOCK_HW → [BLOCK_C]

    # ---- write output [BLOCK_C] -------------------------------------------
    out = hs * (sums / HW)
    tl.store(out_ptr + pid_b * C + c_offs, out, mask=c_mask)


@torch.fx.wrap
def fused_hs_mul_avgpool(conv2d_out, in2):
    B  = in2.shape[0]
    C  = in2.shape[1]
    H  = in2.shape[2]
    W  = in2.shape[3]
    HW = H * W

    out = torch.empty((B, C), dtype=in2.dtype, device=in2.device)

    # Choose tile sizes based on the spatial size:
    #   HW ≤ 64  → compact tile (no masking, exact fit)
    #   HW > 64  → wider tile  (one pass, some masking)
    if HW <= 64:
        BLOCK_C_VAL  = 32
        BLOCK_HW_VAL = 64
        NW           = 4
    else:
        BLOCK_C_VAL  = 16
        BLOCK_HW_VAL = 256
        NW           = 4

    grid = (B * triton.cdiv(C, BLOCK_C_VAL),)

    fused_hs_mul_avgpool_kernel[grid](
        conv2d_out,
        in2,
        out,
        B, C, HW,
        BLOCK_C=BLOCK_C_VAL, BLOCK_HW=BLOCK_HW_VAL,
        num_warps=NW,
    )

    return out


def replacement_func():
    return fused_hs_mul_avgpool