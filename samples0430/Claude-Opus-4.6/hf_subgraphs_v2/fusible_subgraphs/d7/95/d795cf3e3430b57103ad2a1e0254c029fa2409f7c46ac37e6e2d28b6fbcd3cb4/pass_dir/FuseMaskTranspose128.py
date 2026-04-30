import torch
from torch import device
import triton
import triton.language as tl


def pattern(in_0):
    tmp_0 = torch.zeros((1, 133, 133), device=device(type='cuda', index=0))
    tmp_1 = tmp_0[(slice(None, None, None), slice(-5, None, None), slice(None, None, None))]
    tmp_2 = tmp_1.fill_(1)
    tmp_3 = tmp_0[(slice(None, None, None), slice(None, None, None), slice(-5, None, None))]
    tmp_4 = tmp_3.fill_(1)
    tmp_5 = in_0.reshape(1, 19, 7, 19, 7, 128)
    tmp_6 = tmp_5.transpose(2, 3)
    tmp_7 = tmp_0.reshape(1, 19, 7, 19, 7)
    tmp_8 = tmp_7.transpose(2, 3)
    tmp_9 = tmp_8.reshape(1, 361, 49)
    tmp_10 = tmp_9.unsqueeze(2)
    tmp_11 = tmp_9.unsqueeze(3)
    tmp_12 = tmp_10 - tmp_11
    tmp_13 = tmp_12 != 0
    tmp_14 = tmp_12.masked_fill(tmp_13, -1000.0)
    tmp_15 = tmp_12 == 0
    tmp_16 = tmp_14.masked_fill(tmp_15, 0.0)
    return (tmp_16, tmp_6)


def replacement_args(in_0):
    return (in_0,)


@triton.jit
def attn_mask_kernel_128(out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    valid = offsets < n_elements

    # Decode flat index to (w, i, j) in shape (361, 49, 49)
    j = offsets % 49
    temp = offsets // 49
    i = temp % 49
    w = temp // 49

    # Compute window indices
    a = w // 19   # row window index (0..18)
    c = w % 19    # col window index (0..18)

    # Mask value for position i
    bi = i // 7
    di = i % 7
    mask_i = ((a == 18) & (bi >= 2)) | ((c == 18) & (di >= 2))

    # Mask value for position j
    bj = j // 7
    dj = j % 7
    mask_j = ((a == 18) & (bj >= 2)) | ((c == 18) & (dj >= 2))

    # Output: 0 where mask values are equal, -1000 where different
    result = tl.where(mask_i == mask_j, 0.0, -1000.0)

    tl.store(out_ptr + offsets, result, mask=valid)


@torch.fx.wrap
def fused_mask_and_transpose_128(in_0):
    # Generate attention mask using Triton kernel
    n_elements = 361 * 49 * 49  # 866761
    mask_out = torch.empty((1, 361, 49, 49), dtype=torch.float32, device=in_0.device)

    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    attn_mask_kernel_128[grid](mask_out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Reshape and transpose input (view operations, no data copy)
    tmp_6 = in_0.reshape(1, 19, 7, 19, 7, 128).transpose(2, 3)

    return (mask_out, tmp_6)


def replacement_func():
    return fused_mask_and_transpose_128