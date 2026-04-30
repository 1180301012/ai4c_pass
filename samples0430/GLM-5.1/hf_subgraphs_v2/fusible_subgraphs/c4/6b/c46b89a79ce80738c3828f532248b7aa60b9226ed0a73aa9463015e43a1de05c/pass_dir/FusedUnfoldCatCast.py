import torch
import triton
import triton.language as tl


def pattern(in_0, in_1, in_2):
    tmp_0 = torch.nn.functional.unfold(in_1, kernel_size = (384, 384), stride = (192, 192))
    tmp_1 = tmp_0.permute(2, 0, 1)
    tmp_2 = tmp_1.reshape(-1, 3, 384, 384)
    tmp_3 = torch.nn.functional.unfold(in_2, kernel_size = (384, 384), stride = (288, 288))
    tmp_4 = tmp_3.permute(2, 0, 1)
    tmp_5 = tmp_4.reshape(-1, 3, 384, 384)
    tmp_6 = torch.cat([tmp_5, tmp_2, in_0], dim = 0)
    tmp_7 = tmp_6.to(dtype = torch.float16)
    return (tmp_7,)


def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)


@triton.jit
def fused_unfold_cat_cast_kernel(
    in_0_ptr, in_1_ptr, in_2_ptr, out_ptr,
    IN0_H: tl.constexpr, IN0_W: tl.constexpr,
    IN1_H: tl.constexpr, IN1_W: tl.constexpr,
    IN2_H: tl.constexpr, IN2_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_PH: tl.constexpr, OUT_PW: tl.constexpr,
    SH1: tl.constexpr, SW1: tl.constexpr,
    SH2: tl.constexpr, SW2: tl.constexpr,
    NP2: tl.constexpr, NP1: tl.constexpr,
    LW2: tl.constexpr,
    LW1: tl.constexpr,
):
    # 2D grid: (b, ch) where b is patch index, ch is channel*row index
    b = tl.program_id(0)
    ch = tl.program_id(1)
    
    # Decode channel and row from ch
    c = ch // OUT_PH
    h = ch % OUT_PH
    
    # Column offsets for this row
    w_range = tl.arange(0, OUT_PW)
    
    # Output offset: element at (b, c, h, w) in [35, 3, 384, 384] output
    out_offset = b * OUT_C * OUT_PH * OUT_PW + c * OUT_PH * OUT_PW + h * OUT_PW + w_range
    
    # Branch based on which source image this patch comes from
    # Patches 0-24 (b < 25): from in_2 at stride (288, 288)
    # Patches 25-33 (b < 34): from in_1 at stride (192, 192)
    # Patch 34 (b >= 34): from in_0 directly
    
    if b < NP2:
        # Extract patch from in_2
        # Patch b maps to 2D grid position: row = b // LW2, col = b % LW2
        pr = b // LW2
        pc = b % LW2
        src_h = pr * SH2 + h
        src_w = pc * SW2 + w_range
        # in_2 has shape [1, 3, 1536, 1536], batch dim stride = C*H*W
        offset = c * IN2_H * IN2_W + src_h * IN2_W + src_w
        val = tl.load(in_2_ptr + offset)
    else:
        if b < NP2 + NP1:
            # Extract patch from in_1
            b1 = b - NP2
            pr = b1 // LW1
            pc = b1 % LW1
            src_h = pr * SH1 + h
            src_w = pc * SW1 + w_range
            # in_1 has shape [1, 3, 768, 768]
            offset = c * IN1_H * IN1_W + src_h * IN1_W + src_w
            val = tl.load(in_1_ptr + offset)
        else:
            # Copy from in_0 directly
            # in_0 has shape [1, 3, 384, 384]
            offset = c * IN0_H * IN0_W + h * IN0_W + w_range
            val = tl.load(in_0_ptr + offset)
    
    # Store to output (automatic type conversion to float16)
    tl.store(out_ptr + out_offset, val)


@torch.fx.wrap
def fused_unfold_cat_cast(in_0, in_1, in_2):
    # Constants from the model definition
    OUT_C = 3
    OUT_PH = 384
    OUT_PW = 384
    
    IN0_H = 384
    IN0_W = 384
    IN1_H = 768
    IN1_W = 768
    IN2_H = 1536
    IN2_W = 1536
    
    # Unfold strides
    SH1 = 192
    SW1 = 192
    SH2 = 288
    SW2 = 288
    
    # Number of patches from each source
    # in_2: (1536-384)/288 + 1 = 5 patches per dimension, total 5*5=25
    NP2 = 25
    # in_1: (768-384)/192 + 1 = 3 patches per dimension, total 3*3=9
    NP1 = 9
    # in_0: 1 patch (the whole image)
    
    # Grid dimensions for patch enumeration (row-major: L = row * L_W + col)
    LW2 = 5  # number of columns in in_2 patch grid
    LW1 = 3  # number of columns in in_1 patch grid
    
    OUT_B = 35  # NP2 + NP1 + 1 = 25 + 9 + 1
    
    # Allocate output tensor (float16, matching the .to(dtype=torch.float16) operation)
    out = torch.empty((OUT_B, OUT_C, OUT_PH, OUT_PW), dtype=torch.float16, device=in_0.device)
    
    # 2D grid: each program handles one row of one patch
    # grid[0] = number of patches (35)
    # grid[1] = number of (channel, row) pairs (3 * 384 = 1152)
    grid = (OUT_B, OUT_C * OUT_PH)
    
    fused_unfold_cat_cast_kernel[grid](
        in_0_ptr=in_0, in_1_ptr=in_1, in_2_ptr=in_2, out_ptr=out,
        IN0_H=IN0_H, IN0_W=IN0_W,
        IN1_H=IN1_H, IN1_W=IN1_W,
        IN2_H=IN2_H, IN2_W=IN2_W,
        OUT_C=OUT_C, OUT_PH=OUT_PH, OUT_PW=OUT_PW,
        SH1=SH1, SW1=SW1,
        SH2=SH2, SW2=SW2,
        NP2=NP2, NP1=NP1,
        LW2=LW2, LW1=LW1,
    )
    
    return (out,)


def replacement_func():
    return fused_unfold_cat_cast