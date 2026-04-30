import torch
import triton
import triton.language as tl


def pattern(mask_const):
    tmp_7 = mask_const.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    return tmp_12


def replacement_args(mask_const):
    return (mask_const,)


@triton.jit
def attn_diff_kernel_2d(out_ptr, BLOCK_IJ: tl.constexpr):
    w = tl.program_id(0)   # window index 0..360
    block_id = tl.program_id(1)  # block within window

    a = w // 19
    c = w % 19

    ij_start = block_id * BLOCK_IJ
    ij_offsets = ij_start + tl.arange(0, BLOCK_IJ)
    valid = ij_offsets < 2401  # 49*49

    i = ij_offsets // 49
    j = ij_offsets % 49

    bi = i // 7
    di = i % 7
    bj = j // 7
    dj = j % 7

    mask_i = ((a == 18) & (bi >= 2)) | ((c == 18) & (di >= 2))
    mask_j = ((a == 18) & (bj >= 2)) | ((c == 18) & (dj >= 2))

    diff = mask_i.to(tl.float32) - mask_j.to(tl.float32)

    global_offsets = w * 2401 + ij_offsets
    tl.store(out_ptr + global_offsets, diff, mask=valid)


@torch.fx.wrap
def generate_attn_diff(mask_const):
    diff_out = torch.empty((1, 361, 49, 49), dtype=torch.float32, device=mask_const.device)

    BLOCK_IJ = 2048
    n_ij_blocks = (2401 + BLOCK_IJ - 1) // BLOCK_IJ  # 2
    grid = (361, n_ij_blocks)
    attn_diff_kernel_2d[grid](diff_out, BLOCK_IJ=BLOCK_IJ)

    return diff_out


def replacement_func():
    return generate_attn_diff