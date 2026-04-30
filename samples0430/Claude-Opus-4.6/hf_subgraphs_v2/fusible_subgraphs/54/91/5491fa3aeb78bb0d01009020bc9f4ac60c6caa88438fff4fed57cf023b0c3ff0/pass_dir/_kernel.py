import torch
import triton
import triton.language as tl


@triton.jit
def fused_relu6_gap_kernel(
    input_ptr,
    output_ptr,
    total_channels,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    ch_start = pid * BLOCK_C
    ch_offsets = tl.arange(0, BLOCK_C)
    hw_offsets = tl.arange(0, BLOCK_HW)
    ch_idx = ch_start + ch_offsets
    ch_mask = ch_idx < total_channels
    # 2D load: [BLOCK_C, BLOCK_HW] - memory is contiguous for consecutive channels
    offsets = ch_idx[:, None] * HW + hw_offsets[None, :]
    hw_mask = hw_offsets[None, :] < HW
    full_mask = ch_mask[:, None] & hw_mask
    x = tl.load(input_ptr + offsets, mask=full_mask, other=0.0)
    x = tl.maximum(x, 0.0)
    x = tl.minimum(x, 6.0)
    sum_val = tl.sum(x, axis=1)
    avg_val = sum_val / HW
    tl.store(output_ptr + ch_idx, avg_val, mask=ch_mask)


@torch.fx.wrap
def fused_relu6_gap_flatten(in_0):
    B = in_0.shape[0]
    C = in_0.shape[1]
    HW = in_0.shape[2] * in_0.shape[3]
    total_channels = B * C
    out = torch.empty((B, C), dtype=in_0.dtype, device=in_0.device)
    BLOCK_HW = triton.next_power_of_2(HW)
    if BLOCK_HW < 16:
        BLOCK_HW = 16
    # Process multiple channels per program - target ~512 elements per program
    BLOCK_C = max(1, 512 // BLOCK_HW)
    num_programs = (total_channels + BLOCK_C - 1) // BLOCK_C
    fused_relu6_gap_kernel[(num_programs,)](
        in_0, out, total_channels, HW,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW, num_warps=4
    )
    return out