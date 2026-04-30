import torch
import triton
import triton.language as tl


def pattern(x):
    tmp_0 = x * 0.1767766952966369
    tmp_1 = tmp_0.softmax(dim=-1)
    tmp_2 = tmp_1.transpose(-2, -1)
    return tmp_2


def replacement_args(x):
    return (x,)


@triton.jit
def fused_scale_softmax_transpose_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    MN,
    scale,
    TILE_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    tile_id = tl.program_id(0)
    bh = tl.program_id(1)

    m_start = tile_id * TILE_M
    offs_m = m_start + tl.arange(0, TILE_M)
    offs_n = tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Load [TILE_M, BLOCK_N] block (coalesced reads: N is inner dim)
    in_mask = mask_m[:, None] & mask_n[None, :]
    in_offset = bh * MN + offs_m[:, None] * N + offs_n[None, :]
    x = tl.load(input_ptr + in_offset, mask=in_mask, other=-float('inf'))

    # Scale + softmax along axis 1 (in float32)
    x_f32 = x.to(tl.float32) * scale
    x_max = tl.max(x_f32, axis=1)[:, None]
    x_f32 = x_f32 - x_max
    x_exp = tl.exp(x_f32)
    x_sum = tl.sum(x_exp, axis=1)[:, None]
    softmax_out = x_exp / x_sum
    result = softmax_out.to(x.dtype)

    # Write transposed [BLOCK_N, TILE_M] to output (M is inner dim -> coalesced)
    out_mask = mask_n[:, None] & mask_m[None, :]
    out_offset = bh * MN + offs_n[:, None] * M + offs_m[None, :]
    tl.store(output_ptr + out_offset, tl.trans(result), mask=out_mask)


@torch.fx.wrap
def fused_scale_softmax_transpose(x):
    shape = x.shape
    B_H = 1
    for i in range(len(shape) - 2):
        B_H *= shape[i]
    M = shape[-2]
    N = shape[-1]

    # Output: transposed shape [B, H, N, M], contiguous
    out_shape = list(shape[:-2]) + [N, M]
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)

    TILE_M = 32
    num_tiles = (M + TILE_M - 1) // TILE_M
    grid = (num_tiles, B_H)

    fused_scale_softmax_transpose_kernel[grid](
        x, out, M, N, M * N, 0.1767766952966369,
        TILE_M=TILE_M, BLOCK_N=512, num_warps=8
    )

    return out


def replacement_func():
    return fused_scale_softmax_transpose