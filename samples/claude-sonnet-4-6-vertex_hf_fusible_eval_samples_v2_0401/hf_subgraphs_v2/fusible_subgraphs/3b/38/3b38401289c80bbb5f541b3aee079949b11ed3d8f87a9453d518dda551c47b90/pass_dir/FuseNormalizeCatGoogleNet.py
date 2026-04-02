import torch
import triton
import triton.language as tl


@triton.jit
def fused_normalize_cat_kernel(
    in0_ptr, in1_ptr, out_ptr,
    B, HW, C0,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fuses: channel-select, unsqueeze, scale+shift for 3 channels, and cat.

    in_1 : [B, 1,  H, W]  -> out channel 0:  v * 0.458 - 0.03
    in_0 : [B, C0, H, W]  -> out channel 1:  in_0[:,1] * 0.448 - 0.088
                           -> out channel 2:  in_0[:,2] * 0.45  - 0.188
    out  : [B, 3,  H, W]
    """
    pid_b = tl.program_id(1)   # batch index
    pid_s = tl.program_id(0)   # spatial block index

    offsets = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask    = offsets < HW

    # ---- channel 0: from in_1[pid_b, 0, :, :] ----
    # in_1 strides: (HW, HW, W, 1)  -> element [b,0,h,w] = b*HW + h*W + w
    v1   = tl.load(in1_ptr + pid_b * HW + offsets, mask=mask, other=0.0).to(tl.float32)
    out0 = v1 * 0.458 + (-0.030000000000000027)

    # ---- channel 1: from in_0[pid_b, 1, :, :] ----
    # in_0 strides: (C0*HW, HW, W, 1)
    in0_c1_base = (pid_b * C0 + 1) * HW
    v1c  = tl.load(in0_ptr + in0_c1_base + offsets, mask=mask, other=0.0).to(tl.float32)
    out1 = v1c * 0.448 + (-0.08799999999999997)

    # ---- channel 2: from in_0[pid_b, 2, :, :] ----
    in0_c2_base = (pid_b * C0 + 2) * HW
    v2c  = tl.load(in0_ptr + in0_c2_base + offsets, mask=mask, other=0.0).to(tl.float32)
    out2 = v2c * 0.45 + (-0.18799999999999994)

    # ---- store to out[pid_b, 0/1/2, :, :] ----
    # out strides: (3*HW, HW, W, 1)
    out_base = pid_b * 3 * HW
    tl.store(out_ptr + out_base            + offsets, out0, mask=mask)
    tl.store(out_ptr + out_base +     HW   + offsets, out1, mask=mask)
    tl.store(out_ptr + out_base + 2 * HW   + offsets, out2, mask=mask)


@torch.fx.wrap
def fused_normalize_cat(in_0, in_1):
    B  = in_0.shape[0]
    C0 = in_0.shape[1]
    H  = in_0.shape[2]
    W  = in_0.shape[3]
    HW = H * W

    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    # Choose BLOCK_SIZE for max SM occupancy:
    #   HW=256  → BLOCK_SIZE=64,  num_warps=2  (4 blocks/batch)
    #   HW=1024 → BLOCK_SIZE=128, num_warps=2  (8 blocks/batch)
    #   HW=50176 → BLOCK_SIZE=1024, num_warps=4 (49 blocks/batch)
    if HW <= 256:
        BLOCK_SIZE = 64
        num_warps  = 2
    elif HW <= 1024:
        BLOCK_SIZE = 128
        num_warps  = 2
    else:
        BLOCK_SIZE = 1024
        num_warps  = 4

    # 2-D grid: (spatial_blocks, batch)
    grid = ((HW + BLOCK_SIZE - 1) // BLOCK_SIZE, B)

    fused_normalize_cat_kernel[grid](
        in_0, in_1, out,
        B, HW, C0,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )

    return out


# ---------------------------------------------------------------------------
# Pattern / replacement wiring
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_1  = in_1 * 0.458
    tmp_2  = tmp_1 + -0.030000000000000027
    tmp_3  = in_0[(slice(None, None, None), 1)]
    tmp_4  = torch.unsqueeze(tmp_3, 1)
    tmp_5  = tmp_4 * 0.448
    tmp_6  = tmp_5 + -0.08799999999999997
    tmp_7  = in_0[(slice(None, None, None), 2)]
    tmp_8  = torch.unsqueeze(tmp_7, 1)
    tmp_9  = tmp_8 * 0.45
    tmp_10 = tmp_9 + -0.18799999999999994
    tmp_11 = torch.cat((tmp_2, tmp_6, tmp_10), 1)
    return tmp_11


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return fused_normalize_cat