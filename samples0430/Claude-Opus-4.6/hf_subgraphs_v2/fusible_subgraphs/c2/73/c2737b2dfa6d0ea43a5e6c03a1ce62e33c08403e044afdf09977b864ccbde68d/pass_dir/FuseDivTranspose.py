import torch
import triton
import triton.language as tl


def pattern(x, divisor):
    tmp_0 = x / divisor
    tmp_1 = tmp_0.transpose(-1, -2)
    return tmp_1


def replacement_args(x, divisor):
    return (x, divisor)


@triton.jit
def div_transpose_kernel(
    input_ptr,
    output_ptr,
    S,
    D,
    SD,
    scale,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_bh = tl.program_id(1)
    pid_sd = tl.program_id(0)

    num_d_blocks = tl.cdiv(D, BLOCK_D)
    pid_s = pid_sd // num_d_blocks
    pid_d = pid_sd % num_d_blocks

    s_offset = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    d_offset = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)

    s_mask = s_offset < S
    d_mask = d_offset < D
    mask = s_mask[:, None] & d_mask[None, :]

    # Input: [BH, S, D] contiguous
    input_base = pid_bh * SD
    input_offsets = s_offset[:, None] * D + d_offset[None, :]
    data = tl.load(input_ptr + input_base + input_offsets, mask=mask, other=0.0)
    data = data * scale

    # Output: [BH, D, S] contiguous
    output_offsets = d_offset[None, :] * S + s_offset[:, None]
    tl.store(output_ptr + input_base + output_offsets, data, mask=mask)


@torch.fx.wrap
def div_transpose(x, divisor):
    B, H, S, D = x.shape
    scale = 1.0 / divisor
    output = torch.empty(B, H, D, S, dtype=x.dtype, device=x.device)
    BH = B * H
    SD = S * D
    BLOCK_S = 128
    BLOCK_D = 8
    num_s_blocks = (S + BLOCK_S - 1) // BLOCK_S
    num_d_blocks = (D + BLOCK_D - 1) // BLOCK_D
    grid = (num_s_blocks * num_d_blocks, BH)
    div_transpose_kernel[grid](
        x, output, S, D, SD, scale,
        BLOCK_S=BLOCK_S, BLOCK_D=BLOCK_D, num_warps=2, num_stages=1
    )
    return output


def replacement_func():
    return div_transpose