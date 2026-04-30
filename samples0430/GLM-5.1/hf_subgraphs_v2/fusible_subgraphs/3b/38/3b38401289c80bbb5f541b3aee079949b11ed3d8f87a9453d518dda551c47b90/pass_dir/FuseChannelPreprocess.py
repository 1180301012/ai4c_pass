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
    return (tmp_11,)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_preprocess_kernel(
    in_0_ptr, in_1_ptr, out_ptr,
    B, C_in, H, W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Output shape [B, 3, H, W]
    # Decompose flat offset into (b, c, h, w)
    w_idx = offsets % W
    hw_idx = offsets // W
    h_idx = hw_idx % H
    chw_idx = hw_idx // H
    c_idx = chw_idx % 3
    b_idx = chw_idx // 3

    HW = H * W

    # Per-channel masks for conditional loading - skip loads for unneeded channels
    mask_ch0 = mask & (c_idx == 0)
    mask_ch1 = mask & (c_idx == 1)
    mask_ch2 = mask & (c_idx == 2)

    # Compute input offsets
    in_1_off = b_idx * HW + h_idx * W + w_idx
    in_0_ch1_off = b_idx * C_in * HW + HW + h_idx * W + w_idx
    in_0_ch2_off = b_idx * C_in * HW + 2 * HW + h_idx * W + w_idx

    # Conditional loads - only load what's needed per channel
    val_in1 = tl.load(in_1_ptr + in_1_off, mask=mask_ch0, other=0.0)
    val_in0_ch1 = tl.load(in_0_ptr + in_0_ch1_off, mask=mask_ch1, other=0.0)
    val_in0_ch2 = tl.load(in_0_ptr + in_0_ch2_off, mask=mask_ch2, other=0.0)

    # Compute scale + shift per channel
    ch0 = val_in1 * 0.458 + (-0.030000000000000027)
    ch1 = val_in0_ch1 * 0.448 + (-0.08799999999999997)
    ch2 = val_in0_ch2 * 0.45 + (-0.18799999999999994)

    # Select correct channel value based on output position
    out = tl.where(c_idx == 0, ch0, tl.where(c_idx == 1, ch1, ch2))

    # Contiguous store - well coalesced
    tl.store(out_ptr + offsets, out, mask=mask)


@torch.fx.wrap
def fused_preprocess(in_0, in_1):
    B = in_0.shape[0]
    C_in = in_0.shape[1]
    H = in_0.shape[2]
    W = in_0.shape[3]

    out = torch.empty((B, 3, H, W), dtype=in_0.dtype, device=in_0.device)

    n_elements = B * 3 * H * W
    grid = ((n_elements + 64 - 1) // 64,)  # Minimum block size for autotune

    fused_preprocess_kernel[grid](
        in_0_ptr=in_0, in_1_ptr=in_1, out_ptr=out,
        B=B, C_in=C_in, H=H, W=W,
        n_elements=n_elements,
    )

    return (out,)


def replacement_func():
    return fused_preprocess