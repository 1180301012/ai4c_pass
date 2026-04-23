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
    out_ptr,       # output tensor [35, 3, 384, 384] float16
    in0_ptr,       # in_0 [1, 3, 384, 384] bfloat16
    in1_ptr,       # in_1 [1, 3, 768, 768] bfloat16
    in2_ptr,       # in_2 [1, 3, 1536, 1536] bfloat16
    n_total,       # total number of output elements = 35*3*384*384
    # Dimensions
    N_PATCHES: tl.constexpr,  # 35
    C: tl.constexpr,          # 3
    KH: tl.constexpr,         # 384
    KW: tl.constexpr,         # 384
    # Source dimensions
    IN1_H: tl.constexpr,      # 768
    IN1_W: tl.constexpr,      # 768
    IN2_H: tl.constexpr,      # 1536
    IN2_W: tl.constexpr,      # 1536
    IN0_H: tl.constexpr,      # 384
    IN0_W: tl.constexpr,      # 384
    # Strides for unfold
    STRIDE1: tl.constexpr,    # 192
    STRIDE2: tl.constexpr,    # 288
    # Number of patches from each source
    N_PATCHES_IN2: tl.constexpr,  # 25
    N_PATCHES_IN1: tl.constexpr,  # 9
    N_PATCHES_IN0: tl.constexpr,  # 1
    # Grid dims for patches
    GRID2_H: tl.constexpr,   # 5
    GRID2_W: tl.constexpr,   # 5
    GRID1_H: tl.constexpr,   # 3
    GRID1_W: tl.constexpr,   # 3
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_total

    # Decompose offset into (patch, channel, kh, kw)
    # offset = p * (C * KH * KW) + c * (KH * KW) + kh * KW + kw
    one_patch_size = C * KH * KW
    one_channel_size = KH * KW

    p = offsets // one_patch_size
    remainder = offsets - p * one_patch_size
    c = remainder // one_channel_size
    remainder2 = remainder - c * one_channel_size
    kh = remainder2 // KW
    kw = remainder2 - kh * KW

    # Determine source and compute source coordinates
    # For in_2 patches (p < 25): stride=288, grid=5x5
    # source_y = (p // 5) * 288 + kh
    # source_x = (p % 5) * 288 + kw
    src_y_in2 = (p // GRID2_W) * STRIDE2 + kh
    src_x_in2 = (p % GRID2_W) * STRIDE2 + kw

    # For in_1 patches (25 <= p < 34): stride=192, grid=3x3
    p1 = p - N_PATCHES_IN2
    src_y_in1 = (p1 // GRID1_W) * STRIDE1 + kh
    src_x_in1 = (p1 % GRID1_W) * STRIDE1 + kw

    # For in_0 patch (p == 34):
    src_y_in0 = kh
    src_x_in0 = kw

    # Compute source linear offsets
    # in_2: shape [1, 3, 1536, 1536], offset = c * 1536 * 1536 + src_y * 1536 + src_x
    in2_offset = c * IN2_H * IN2_W + src_y_in2 * IN2_W + src_x_in2

    # in_1: shape [1, 3, 768, 768], offset = c * 768 * 768 + src_y * 768 + src_x
    in1_offset = c * IN1_H * IN1_W + src_y_in1 * IN1_W + src_x_in1

    # in_0: shape [1, 3, 384, 384], offset = c * 384 * 384 + src_y * 384 + src_x
    in0_offset = c * IN0_H * IN0_W + src_y_in0 * IN0_W + src_x_in0

    # Load from appropriate source based on patch index
    # Using conditional loads
    val = tl.where(p < N_PATCHES_IN2,
                   tl.load(in2_ptr + in2_offset, mask=mask & (p < N_PATCHES_IN2)),
                   tl.where(p < N_PATCHES_IN2 + N_PATCHES_IN1,
                            tl.load(in1_ptr + in1_offset, mask=mask & (p >= N_PATCHES_IN2) & (p < N_PATCHES_IN2 + N_PATCHES_IN1)),
                            tl.load(in0_ptr + in0_offset, mask=mask & (p >= N_PATCHES_IN2 + N_PATCHES_IN1))))

    # Cast to float16 and store
    tl.store(out_ptr + offsets, val.to(tl.float16), mask=mask)


@torch.fx.wrap
def fused_unfold_cat_cast(in_0, in_1, in_2):
    # Output shape: [35, 3, 384, 384] float16
    N_PATCHES = 35
    C = 3
    KH = 384
    KW = 384
    n_total = N_PATCHES * C * KH * KW  # 16,257,024

    out = torch.empty(N_PATCHES, C, KH, KW, dtype=torch.float16, device=in_0.device)

    BLOCK_SIZE = 1024
    num_programs = (n_total + BLOCK_SIZE - 1) // BLOCK_SIZE

    fused_unfold_cat_cast_kernel[(num_programs,)](
        out_ptr=out,
        in0_ptr=in_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        n_total=n_total,
        N_PATCHES=35,
        C=3,
        KH=384,
        KW=384,
        IN1_H=768,
        IN1_W=768,
        IN2_H=1536,
        IN2_W=1536,
        IN0_H=384,
        IN0_W=384,
        STRIDE1=192,
        STRIDE2=288,
        N_PATCHES_IN2=25,
        N_PATCHES_IN1=9,
        N_PATCHES_IN0=1,
        GRID2_H=5,
        GRID2_W=5,
        GRID1_H=3,
        GRID1_W=3,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return (out,)


def replacement_func():
    return fused_unfold_cat_cast