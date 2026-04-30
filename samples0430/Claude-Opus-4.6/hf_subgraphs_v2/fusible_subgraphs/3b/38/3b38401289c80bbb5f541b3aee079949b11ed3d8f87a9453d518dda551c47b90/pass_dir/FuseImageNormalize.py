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


@triton.jit
def fused_normalize_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    C0_HW,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    # 3D grid: (3, B, ceil(HW/BLOCK_SIZE))
    # All threads in a program share the same channel - no warp divergence
    channel_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    sp_block = tl.program_id(2)

    sp_start = sp_block * BLOCK_SIZE
    offsets = sp_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < HW

    # Output: [B, 3, H, W] layout
    out_off = (batch_idx * 3 + channel_idx) * HW + offsets

    is_ch0 = (channel_idx == 0)
    is_ch1 = (channel_idx == 1)

    # Source offsets
    in_1_off = batch_idx * HW + offsets
    in_0_off = batch_idx * C0_HW + channel_idx * HW + offsets

    # Load from appropriate source
    val_from_in1 = tl.load(in_1_ptr + in_1_off, mask=mask & is_ch0, other=0.0)
    val_from_in0 = tl.load(in_0_ptr + in_0_off, mask=mask & (~is_ch0), other=0.0)

    val = tl.where(is_ch0, val_from_in1, val_from_in0)

    # Apply per-channel scale and bias
    scale = tl.where(is_ch0, 0.458, tl.where(is_ch1, 0.448, 0.45))
    bias = tl.where(is_ch0, -0.030000000000000027,
                    tl.where(is_ch1, -0.08799999999999997, -0.18799999999999994))

    result = val * scale + bias

    tl.store(out_ptr + out_off, result, mask=mask)


@torch.fx.wrap
def fused_normalize(in_0, in_1):
    B, C0, H, W = in_0.shape
    HW = H * W
    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)
    fused_normalize_kernel[(3, B, (HW + 1023) // 1024)](
        in_0, in_1, out, C0 * HW, HW, BLOCK_SIZE=1024)
    return out


def replacement_func():
    return fused_normalize