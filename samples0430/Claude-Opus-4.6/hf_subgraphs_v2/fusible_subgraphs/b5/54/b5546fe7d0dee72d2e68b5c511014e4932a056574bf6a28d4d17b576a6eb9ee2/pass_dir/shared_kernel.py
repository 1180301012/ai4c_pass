import torch
import triton
import triton.language as tl


@triton.jit
def fused_crpe_kernel(
    conv2d_ptr, in_2_ptr, in_3_ptr, in_4_ptr, in_6_ptr, out_ptr,
    C2, C3, H, W, K, S,
    scale,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    total_channels = 8 * K
    s = offsets // total_channels
    hk = offsets % total_channels
    h = hk // K
    k = hk % K

    in_4_idx = h * (S + 1) * K + s * K + k
    in_4_val = tl.load(in_4_ptr + in_4_idx, mask=mask, other=0.0)
    result = scale * in_4_val

    s_gt_0 = s > 0
    spatial_idx = tl.maximum(s - 1, 0)
    channel = hk
    hw = H * W

    in_6_idx = h * S * K + spatial_idx * K + k
    in_6_val = tl.load(in_6_ptr + in_6_idx, mask=mask & s_gt_0, other=0.0)

    in_2_mask = mask & s_gt_0 & (channel < C2)
    in_2_idx = channel * hw + spatial_idx
    in_2_val = tl.load(in_2_ptr + in_2_idx, mask=in_2_mask, other=0.0)

    in_3_mask = mask & s_gt_0 & (channel >= C2) & (channel < C2 + C3)
    in_3_channel = tl.maximum(channel - C2, 0)
    in_3_idx = in_3_channel * hw + spatial_idx
    in_3_val = tl.load(in_3_ptr + in_3_idx, mask=in_3_mask, other=0.0)

    conv_mask = mask & s_gt_0 & (channel >= C2 + C3)
    conv_channel = tl.maximum(channel - C2 - C3, 0)
    conv_idx = conv_channel * hw + spatial_idx
    conv_val = tl.load(conv2d_ptr + conv_idx, mask=conv_mask, other=0.0)

    cat_val = tl.where(channel < C2, in_2_val,
                       tl.where(channel < C2 + C3, in_3_val, conv_val))

    mul_val = in_6_val * cat_val
    result = result + tl.where(s_gt_0, mul_val, tl.zeros_like(mul_val))

    tl.store(out_ptr + offsets, result, mask=mask)


@torch.fx.wrap
def fused_crpe(conv2d, in_2, in_3, in_4, in_6):
    C2 = in_2.shape[1]
    C3 = in_3.shape[1]
    H = in_2.shape[2]
    W = in_2.shape[3]
    K = in_4.shape[3]
    S = in_6.shape[2]
    scale = 1.0 / (float(K) ** 0.5)

    total_elements = (S + 1) * 8 * K
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    out = torch.empty((1, S + 1, 8 * K), dtype=in_4.dtype, device=in_4.device)

    fused_crpe_kernel[(num_programs,)](
        conv2d, in_2, in_3, in_4, in_6, out,
        C2, C3, H, W, K, S,
        scale,
        total_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return out