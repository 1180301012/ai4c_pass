import torch
import triton
import triton.language as tl


def pattern(in_0, in_1):
    tmp_0 = in_0.to(torch.float32)
    tmp_1 = in_1 * tmp_0
    tmp_2 = torch.sum(tmp_1, 1)
    tmp_3 = tmp_0.sum(1)
    tmp_4 = torch.clamp(tmp_3, min=1e-09)
    tmp_5 = tmp_2 / tmp_4
    tmp_6 = torch.cat([tmp_5], 1)
    return tmp_6


def replacement_args(in_0, in_1):
    return (in_0, in_1)


@triton.jit
def masked_mean_pool_kernel(
    in_0_ptr,
    in_1_ptr,
    out_ptr,
    S: tl.constexpr,
    H: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)
    b = tl.program_id(1)

    h_offsets = pid * BLOCK_H + tl.arange(0, BLOCK_H)  # [BLOCK_H]
    s_offsets = tl.arange(0, BLOCK_S)  # [BLOCK_S]

    h_mask = h_offsets < H
    s_mask = s_offsets < S

    # 2D offsets: [BLOCK_S, BLOCK_H]
    base = b * S * H
    offsets_2d = base + s_offsets[:, None] * H + h_offsets[None, :]
    mask_2d = s_mask[:, None] & h_mask[None, :]

    # Load 2D tiles
    mask_vals = tl.load(in_0_ptr + offsets_2d, mask=mask_2d, other=0).to(tl.float32)
    hidden_vals = tl.load(in_1_ptr + offsets_2d, mask=mask_2d, other=0.0).to(tl.float32)

    # Compute numerator and denominator via reduction along axis=0
    num = tl.sum(hidden_vals * mask_vals, axis=0)  # [BLOCK_H]
    den = tl.sum(mask_vals, axis=0)  # [BLOCK_H]

    # Clamp and divide
    den = tl.maximum(den, 1e-9)
    result = num / den

    # Store output
    out_offset = b * H + h_offsets
    tl.store(out_ptr + out_offset, result, mask=h_mask)


@torch.fx.wrap
def masked_mean_pool(in_0, in_1):
    B = in_0.shape[0]
    S = in_0.shape[1]
    H = in_0.shape[2]

    out = torch.empty((B, H), dtype=torch.float32, device=in_0.device)

    BLOCK_H = 64
    BLOCK_S = 16
    num_h_blocks = (H + BLOCK_H - 1) // BLOCK_H
    grid = (num_h_blocks, B)

    masked_mean_pool_kernel[grid](
        in_0, in_1, out,
        S, H,
        BLOCK_H=BLOCK_H,
        BLOCK_S=BLOCK_S,
        num_warps=2,
        num_stages=1,
    )

    return out


def replacement_func():
    return masked_mean_pool