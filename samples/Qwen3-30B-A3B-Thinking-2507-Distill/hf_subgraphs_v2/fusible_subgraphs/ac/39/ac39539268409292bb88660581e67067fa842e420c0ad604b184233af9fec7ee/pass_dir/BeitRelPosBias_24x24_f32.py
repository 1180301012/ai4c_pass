"""
BEiT relative position bias for 24×24 grid (beit-base-patch16-384 float32).
offset=23, scale=47  →  N = 24*24+1 = 577

Uses comma-indexing notation (tmp_12[...,...,...] vs tmp_7[(...,...,...)]).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _pos_bias_24x24_f32_kernel(out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N

    row_off = tl.arange(0, BLOCK)
    row_mask = row_off < N

    r_off = tl.where(row_mask, (row_off.to(tl.int64) - 23) * 47, 0)
    c_off = tl.where(mask, (offs.to(tl.int64) - 23) * 47, 0)

    result = r_off[:, None] + c_off[None, :]
    result = tl.where(row_mask[:, None] & mask[None, :], result,
                      tl.zeros([BLOCK, BLOCK], dtype=tl.int64))
    tl.store(out_ptr + tl.arange(0, BLOCK)[:, None] * N + tl.arange(0, BLOCK)[None, :],
             result, mask=row_mask[:, None] & mask[None, :])


@triton.jit
def _cat_2d_24x24_f32_kernel(in1_ptr, in0_ptr, out_ptr,
                              M1, M0, D, BLOCK_D: tl.constexpr):
    row = tl.program_id(0)
    col_offs = tl.arange(0, BLOCK_D)
    col_mask = col_offs < D

    is_in0 = row >= M1
    local_row = tl.where(is_in0, row - M1, row)

    v1 = tl.load(in1_ptr + row * D + col_offs,
                 mask=(row < M1) & col_mask, other=0)
    v0 = tl.load(in0_ptr + local_row * D + col_offs,
                 mask=(row >= M1) & col_mask, other=0)
    v = tl.where(is_in0, v0, v1)
    tl.store(out_ptr + row * D + col_offs, v, mask=col_mask)


@torch.fx.wrap
def beit_rel_pos_bias_cat_24x24_f32(in_0, in_1):
    N = 577  # 24*24 + 1

    bias = torch.empty((N, N), dtype=torch.int64, device=in_0.device)
    BLOCK_P = 32
    grid_p = ((N + BLOCK_P - 1) // BLOCK_P,)
    _pos_bias_24x24_f32_kernel[grid_p](bias, N, BLOCK=BLOCK_P)

    out0 = torch.empty((in_1.shape[0] + in_0.shape[0], in_1.shape[1]),
                       dtype=in_1.dtype, device=in_1.device)
    M1 = in_1.shape[0]
    M0 = in_0.shape[0]
    D = in_1.shape[1]
    BLOCK_D = 16
    grid_c = (M1 + M0,)
    _cat_2d_24x24_f32_kernel[grid_c](in_1, in_0, out0, M1, M0, D, BLOCK_D=BLOCK_D)

    return (out0, bias.view(-1))


def pattern(in_0, in_1):
    tmp_0 = torch.cat([in_1, in_0])
    tmp_1 = torch.arange(24)
    tmp_2 = torch.arange(24)
    tmp_3 = torch.functional.meshgrid(tmp_1, tmp_2, indexing='ij')
    tmp_4 = tmp_3[0]
    tmp_5 = tmp_3[1]
    tmp_6 = torch.stack((tmp_4, tmp_5))
    tmp_7 = torch.flatten(tmp_6, 1)
    tmp_8 = tmp_7[slice(None, None, None), slice(None, None, None), None]
    tmp_9 = tmp_7[slice(None, None, None), None, slice(None, None, None)]
    tmp_10 = tmp_8 - tmp_9
    tmp_11 = tmp_10.permute(1, 2, 0)
    tmp_12 = tmp_11.contiguous()
    tmp_12[slice(None, None, None), slice(None, None, None), 0] += 23
    tmp_12[slice(None, None, None), slice(None, None, None), 1] += 23
    tmp_12[slice(None, None, None), slice(None, None, None), 0] *= 47
    tmp_22 = torch.zeros(size=(577, 577), dtype=torch.int64)
    tmp_23 = tmp_12.sum(-1)
    tmp_22[slice(1, None, None), slice(1, None, None)] = tmp_23
    tmp_22[0, slice(0, None, None)] = 2209
    tmp_22[slice(0, None, None), 0] = 2210
    tmp_22[0, 0] = 2211
    tmp_28 = tmp_22.view(-1)
    return (tmp_0, tmp_28)


def replacement_args(in_0, in_1):
    return (in_0, in_1)


def replacement_func():
    return beit_rel_pos_bias_cat_24x24_f32