import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_1 = in_1 * 0.458
    tmp_2 = tmp_1 + -0.030000000000000027
    tmp_3 = in_0[(slice(None, None, None), 1)]
    tmp_4 = torch.unsqueeze(tmp_3, 1)
    tmp_5 = tmp_4 * 0.448
    tmp_6 = tmp_5 + -0.08799999999999997
    tmp_7 = in_0[(slice(None, None, None), 2)]
    tmp_8 = torch.unsqueeze(tmp_7, 1)
    tmp_9 = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# -----------------------------------------------------------------------
# 2D grid kernel: grid = (B*3, ceil(HW/BLOCK_SIZE))
#
# Avoids tl.where on pointer types entirely.
# Uses explicit 1-D masks built from scalar tl.where comparisons:
#   oc == 0  → mask = hw_mask & (oc==0) → all-True when oc==0, all-False otherwise
#   oc == 1  → similar
#   oc == 2  → similar
# Then v_in0, v_in0_c2, v_in1 are loaded with their respective masks;
# only the correct one contributes non-zero values.
# -----------------------------------------------------------------------
@triton.jit
def norm_cat_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    C0, HW,
    BLOCK_SIZE: tl.constexpr,
):
    b      = tl.program_id(0)   # batch index
    oc     = tl.program_id(1)   # output channel (0, 1, or 2)
    pid1   = tl.program_id(2)   # hw block

    hw_offs = pid1 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    hw_mask = hw_offs < HW

    in_batch_base = b * C0 * HW

    # Explicit 1-D masks for each output channel (scalar tl.where → 1-bit vector)
    oc0_b = tl.cast(oc == 0, tl.int1)
    oc1_b = tl.cast(oc == 1, tl.int1)
    oc2_b = tl.cast(oc == 2, tl.int1)

    mask_c0 = hw_mask & tl.broadcast_to(oc0_b, [BLOCK_SIZE])
    mask_c1 = hw_mask & tl.broadcast_to(oc1_b, [BLOCK_SIZE])
    mask_c2 = hw_mask & tl.broadcast_to(oc2_b, [BLOCK_SIZE])

    v_c0 = tl.load(in0_ptr + in_batch_base + 1 * HW + hw_offs,
                   mask=mask_c0, other=0.0).to(tl.float32)
    v_c1 = tl.load(in0_ptr + in_batch_base + 2 * HW + hw_offs,
                   mask=mask_c1, other=0.0).to(tl.float32)
    v_c2 = tl.load(in1_ptr + b * HW + hw_offs,
                   mask=mask_c2, other=0.0).to(tl.float32)

    val = v_c0 + v_c1 + v_c2

    scale = tl.where(oc == 0, 0.448, tl.where(oc == 1, 0.45, 0.458))
    bias  = tl.where(oc == 0, -0.08799999999999997,
            tl.where(oc == 1, -0.18799999999999994, -0.030000000000000027))

    out_val = val * scale + bias

    tl.store(out_ptr + (b * 3 + oc) * HW + hw_offs,
             out_val.to(out_ptr.dtype.element_ty),
             mask=hw_mask)


@torch.fx.wrap
def fused_normalize_cat(in_0, in_1):
    B  = in_0.shape[0]
    C0 = in_0.shape[1]
    H  = in_0.shape[2]
    W  = in_0.shape[3]
    HW = H * W

    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    BLOCK_SIZE = 1024
    hw_blocks  = triton.cdiv(HW, BLOCK_SIZE)

    # 3D grid: (B, 3, hw_blocks) — b, oc, hw_block are direct program IDs
    norm_cat_kernel[(B, 3, hw_blocks)](
        in_0, in_1, out,
        C0, HW,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out


def replacement_func():
    return fused_normalize_cat