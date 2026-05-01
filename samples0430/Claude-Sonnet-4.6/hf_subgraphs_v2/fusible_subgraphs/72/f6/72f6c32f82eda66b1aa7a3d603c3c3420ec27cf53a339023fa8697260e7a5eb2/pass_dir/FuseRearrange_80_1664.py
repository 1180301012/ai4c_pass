import torch
import triton
import triton.language as tl


# Grid: (C_MAJOR=8, BLK=4, L=144)
# Each program fuses pad+unfold+unfold+reshape+permute+split+transpose for one (i0, i2, i3) triple.
# conv_out: [1, C_OUT=640, H=16, W=16] (C_MINOR=80, S1=16, S2=64, C_MAJOR=8)
# out1: [8, 4, 16, 144]   <- transposed first split
# out2: [8, 4, 144, 64]   <- second split
@triton.jit
def _fused_rearrange_80_kernel(
    src_ptr,
    out1_ptr,
    out2_ptr,
    H: tl.constexpr,       # 16
    W: tl.constexpr,       # 16
    C_MAJOR: tl.constexpr, # 8
    C_MINOR: tl.constexpr, # 80
    S1: tl.constexpr,      # 16
    S2: tl.constexpr,      # 64
    L: tl.constexpr,       # 144
    KW: tl.constexpr,      # 12
    BLOCK_C: tl.constexpr, # 128 (next power-of-2 >= C_MINOR=80)
):
    i0 = tl.program_id(0)  # [0, C_MAJOR)
    i2 = tl.program_id(1)  # [0, 4)  block index: bh*2 + bw
    i3 = tl.program_id(2)  # [0, L)  spatial index: kh*KW + kw

    # Decode block position
    bh = i2 >> 1
    bw = i2 & 1

    # Decode spatial position within block
    kh = i3 // KW
    kw = i3 % KW

    # Map to conv output coordinates (pad=2 on each side → subtract 2)
    src_h = bh * 8 + kh - 2
    src_w = bw * 8 + kw - 2

    # Check if position is valid (not in padding region)
    valid = (src_h >= 0) & (src_h < H) & (src_w >= 0) & (src_w < W)

    # Channel vector: [0, BLOCK_C)
    j = tl.arange(0, BLOCK_C)
    mask_c = j < C_MINOR
    c = i0 * C_MINOR + j  # absolute channel index

    # Clamp spatial to valid range for safe memory access
    src_h_safe = tl.where(valid, src_h, 0)
    src_w_safe = tl.where(valid, src_w, 0)

    # src is [1, C, H, W] → flat index = c * H * W + h * W + w  (n=0)
    src_idx = c * (H * W) + src_h_safe * W + src_w_safe

    # Load C_MINOR channel values for this spatial position
    vals = tl.load(src_ptr + src_idx, mask=mask_c, other=0.0)
    # Zero out padding positions
    vals = tl.where(valid, vals, 0.0)

    # ---- Write out1: tmp_10[i0, i2, j, i3] for j in [0, S1) ----
    # out1 layout: [C_MAJOR, 4, S1, L] → stride (4*S1*L, S1*L, L, 1)
    mask1 = mask_c & (j < S1)
    out1_idx = i0 * (4 * S1 * L) + i2 * (S1 * L) + j * L + i3
    tl.store(out1_ptr + out1_idx, vals, mask=mask1)

    # ---- Write out2: tmp_9[i0, i2, i3, j-S1] for j in [S1, C_MINOR) ----
    # out2 layout: [C_MAJOR, 4, L, S2] → stride (4*L*S2, L*S2, S2, 1)
    j2 = tl.where(j >= S1, j - S1, 0)  # clamp to avoid negative index
    mask2 = mask_c & (j >= S1)
    out2_idx = i0 * (4 * L * S2) + i2 * (L * S2) + i3 * S2 + j2
    tl.store(out2_ptr + out2_idx, vals, mask=mask2)


def pattern(conv_out):
    """Matches: pad → unfold(2,12,8) → unfold(3,12,8) → reshape(8,80,4,-1)
                → permute(0,2,3,1) → split([16,64],-1) → transpose(-1,-2) on first part"""
    padded = torch.nn.functional.pad(conv_out, [2, 2, 2, 2], 'constant', None)
    unf1 = padded.unfold(2, 12, 8)
    unf2 = unf1.unfold(3, 12, 8)
    reshaped = unf2.reshape(8, 80, 4, -1)
    permuted = reshaped.permute(0, 2, 3, 1)
    split = torch.functional.split(permuted, [16, 64], dim=-1)
    part1 = split[0]
    part2 = split[1]
    transposed = part1.transpose(-1, -2)
    return (transposed, part2)


def replacement_args(conv_out):
    return (conv_out,)


@torch.fx.wrap
def fused_rearrange_80(conv_out):
    C_MAJOR = 8
    C_MINOR = 80
    S1 = 16
    S2 = 64
    H = 16
    W = 16
    KW = 12
    L = 144   # KH * KW = 12 * 12
    BLOCK_C = 128  # next power-of-2 >= C_MINOR=80

    device = conv_out.device
    dtype = conv_out.dtype

    # out1 = tmp_10: [C_MAJOR, 4, S1, L] = [8, 4, 16, 144]
    out1 = torch.empty((C_MAJOR, 4, S1, L), dtype=dtype, device=device)
    # out2 = tmp_9:  [C_MAJOR, 4, L, S2] = [8, 4, 144, 64]
    out2 = torch.empty((C_MAJOR, 4, L, S2), dtype=dtype, device=device)

    grid = (C_MAJOR, 4, L)
    _fused_rearrange_80_kernel[grid](
        conv_out, out1, out2,
        H=H, W=W,
        C_MAJOR=C_MAJOR, C_MINOR=C_MINOR,
        S1=S1, S2=S2,
        L=L, KW=KW,
        BLOCK_C=BLOCK_C,
    )

    return (out1, out2)


def replacement_func():
    return fused_rearrange_80