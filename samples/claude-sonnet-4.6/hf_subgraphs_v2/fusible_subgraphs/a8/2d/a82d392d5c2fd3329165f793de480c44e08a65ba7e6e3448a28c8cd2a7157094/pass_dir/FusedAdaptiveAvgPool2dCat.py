import torch
import triton
import triton.language as tl


# ---------------------------------------------------------------------------
# Pattern: adaptive_avg_pool2d(in_0, (32, 24))  +  cat([pool_out, in_1], dim=1)
#
# in_0 : [B, C0, 64, 48]  →  pool to [B, C0, 32, 24]   (KH=KW=2)
# in_1 : [B, C1, 32, 24]
# out  : [B, C0+C1, 32, 24]
# ---------------------------------------------------------------------------

def pattern(in_0, in_1):
    tmp_0 = torch.nn.functional.adaptive_avg_pool2d(in_0, (32, 24))
    tmp_1 = torch.cat([tmp_0, in_1], dim=1)
    return tmp_1


def replacement_args(in_0, in_1):
    return (in_0, in_1)


# ---------------------------------------------------------------------------
# Single fused kernel — 2-D grid: (B*C_total, ceil(HW/BLOCK_HW))
#
# BLOCK_HW is a tl.constexpr, so each unique value compiles a separate
# kernel variant.  The Python wrapper selects the right variant based on
# the total number of programs (B * C_total):
#
#   B * C_total >= 1024  →  BLOCK_HW=1024  (one full-spatial tile/program,
#                                            max 32 warps/SM, best bandwidth)
#   B * C_total < 1024   →  BLOCK_HW=256   (3 tiles/program, more programs,
#                                            better SM fill for small B)
#
# Pool programs (c < C0): oh/ow vector div/mod only in this branch.
# Copy programs (c >= C0): fully contiguous, zero div/mod overhead.
# ---------------------------------------------------------------------------

@triton.jit
def fused_avgpool_cat_kernel(
    in0_ptr,
    in1_ptr,
    out_ptr,
    B, C0, C1, IH, IW,
    BLOCK_HW: tl.constexpr,
):
    OH = 32
    OW = 24
    HW = 768        # OH * OW
    C_total = C0 + C1

    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    b = pid_bc // C_total
    c = pid_bc % C_total

    hw_start   = pid_hw * BLOCK_HW
    hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
    hw_mask    = hw_offsets < HW

    out_offset = (b * C_total + c) * HW + hw_offsets

    if c < C0:
        # ---- average-pool path (oh/ow only here) ----
        oh   = hw_offsets // OW
        ow   = hw_offsets % OW
        ih0  = oh * 2
        iw0  = ow * 2
        base = (b * C0 + c) * IH * IW + ih0 * IW + iw0

        v00 = tl.load(in0_ptr + base,           mask=hw_mask, other=0.0).to(tl.float32)
        v01 = tl.load(in0_ptr + base + 1,       mask=hw_mask, other=0.0).to(tl.float32)
        v10 = tl.load(in0_ptr + base + IW,      mask=hw_mask, other=0.0).to(tl.float32)
        v11 = tl.load(in0_ptr + base + IW + 1,  mask=hw_mask, other=0.0).to(tl.float32)

        out_val = (v00 + v01 + v10 + v11) * 0.25
        tl.store(out_ptr + out_offset, out_val, mask=hw_mask)

    else:
        # ---- contiguous copy path (zero div/mod) ----
        c1         = c - C0
        in1_offset = (b * C1 + c1) * HW + hw_offsets
        copy_val   = tl.load(in1_ptr + in1_offset, mask=hw_mask, other=0.0)
        tl.store(out_ptr + out_offset, copy_val, mask=hw_mask)


@torch.fx.wrap
def fused_avgpool_cat(in_0, in_1):
    B       = in_0.shape[0]
    C0      = in_0.shape[1]
    C1      = in_1.shape[1]
    IH      = in_0.shape[2]
    IW      = in_0.shape[3]
    OH      = 32
    OW      = 24
    HW      = OH * OW          # 768
    C_total = C0 + C1

    out = torch.empty((B, C_total, OH, OW), dtype=in_0.dtype, device=in_0.device)

    # Select BLOCK_HW based on total program count:
    # Large B (many programs) → BLOCK_HW=1024 for maximum warp occupancy per SM
    # Small B (few programs) → BLOCK_HW=256 for more programs to fill more SMs
    total_programs = B * C_total
    if total_programs >= 1024:
        BLOCK_HW = 1024
    elif total_programs >= 256:
        BLOCK_HW = 512
    else:
        BLOCK_HW = 256

    num_hw_tiles = (HW + BLOCK_HW - 1) // BLOCK_HW
    grid = (total_programs, num_hw_tiles)

    fused_avgpool_cat_kernel[grid](
        in_0, in_1, out,
        B, C0, C1, IH, IW,
        BLOCK_HW=BLOCK_HW,
    )

    return out


def replacement_func():
    return fused_avgpool_cat