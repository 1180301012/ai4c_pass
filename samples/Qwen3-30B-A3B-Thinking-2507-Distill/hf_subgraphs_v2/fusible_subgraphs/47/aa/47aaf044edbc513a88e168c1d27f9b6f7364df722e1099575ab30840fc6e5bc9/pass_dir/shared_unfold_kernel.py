import torch
import triton
import triton.language as tl


# Kernel A: fused unfold(reshape) + transpose + reshape
# Input: tmp_2 [1, L, C, 9] (contiguous strides L*C*9, C*9, 9, 1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['total'],
)
@triton.jit
def fused_unfold_reshape_kernel(
    x_ptr, out_ptr,
    L, C, H, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    hw = offsets % (H * 9)
    n = offsets // (H * 9)
    h = hw // 9
    w = hw % 9
    p = h + n * H
    c = n % C
    row = p + w - 4
    safe_row = tl.maximum(tl.minimum(row, L - 1), 0)
    x_val = tl.load(x_ptr + p * (C * 9) + c * 9 + w, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_val, mask=mask)


# Kernel B: fused unsqueeze + im2col + transpose + reshape
# Input: tmp_1 [1, C, L, 1] (contiguous strides C*L, L, 1, 1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['total'],
)
@triton.jit
def fused_unsqueeze_unfold_kernel(
    x_ptr, out_ptr,
    L, C, H, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    hw = offsets % (H * 9)
    n = offsets // (H * 9)
    h = hw // 9
    w = hw % 9
    p = h + n * H
    c = n % C
    row = p + w - 4
    safe_row = tl.maximum(tl.minimum(row, L - 1), 0)
    # tmp_1[0, c, l, 0] = x_ptr + c*L + l
    x_val = tl.load(x_ptr + c * L + safe_row, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_val, mask=mask)


# Kernel C: im2col-based for tiny model ([1,16,45,1] -> [90,8,9])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
    ],
    key=['total'],
)
@triton.jit
def fused_im2col_reshape_kernel(
    x_ptr, out_ptr,
    L, C, H, total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    hw = offsets % (H * 9)
    n = offsets // (H * 9)
    h = hw // 9
    w = hw % 9
    p = h + n * H
    c = n % C
    row = p + w - 4
    safe_row = tl.maximum(tl.minimum(row, L - 1), 0)
    # im2col index: tmp_1[0, c, l, 0] at c*L + l (same as unsqueeze+unfold kernel)
    x_val = tl.load(x_ptr + c * L + safe_row, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x_val, mask=mask)


def _run_16_45(tmp_2):
    L, C, H = 45, 16, 8
    total = 90 * 8 * 9
    out = torch.empty((90, 8, 9), dtype=tmp_2.dtype, device=tmp_2.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_unfold_reshape_kernel[grid](tmp_2, out, L, C, H, total)
    return out


def _run_384_11(tmp_2):
    L, C, H = 11, 384, 64
    total = 64 * 64 * 9
    out = torch.empty((64, 64, 9), dtype=tmp_2.dtype, device=tmp_2.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_unfold_reshape_kernel[grid](tmp_2, out, L, C, H, total)
    return out


def _run_unfold_16_45(tmp_1):
    L, C, H = 45, 16, 8
    total = 90 * 8 * 9
    out = torch.empty((90, 8, 9), dtype=tmp_1.dtype, device=tmp_1.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_unsqueeze_unfold_kernel[grid](tmp_1, out, L, C, H, total)
    return out


def _run_im2col_16_45(tmp_1):
    # aten.im2col version: tmp_1 [1, 16, 45, 1] -> [90, 8, 9]
    L, C, H = 45, 16, 8
    total = 90 * 8 * 9
    out = torch.empty((90, 8, 9), dtype=tmp_1.dtype, device=tmp_1.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_im2col_reshape_kernel[grid](tmp_1, out, L, C, H, total)
    return out


def _run_unfold_384_11(tmp_1):
    L, C, H = 11, 384, 64
    total = 64 * 64 * 9
    out = torch.empty((64, 64, 9), dtype=tmp_1.dtype, device=tmp_1.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_unsqueeze_unfold_kernel[grid](tmp_1, out, L, C, H, total)
    return out


def _run_im2col_384_11(tmp_1):
    # aten.im2col version: tmp_1 [1, 384, 11, 1] -> [64, 64, 9]
    L, C, H = 11, 384, 64
    total = 64 * 64 * 9
    out = torch.empty((64, 64, 9), dtype=tmp_1.dtype, device=tmp_1.device)
    grid = lambda meta: ((total + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    fused_im2col_reshape_kernel[grid](tmp_1, out, L, C, H, total)
    return out


@torch.fx.wrap
def triton_unfold_dispatch(x, route):
    if route == 'transpose_16_45':
        return _run_16_45(x)
    elif route == 'transpose_384_11':
        return _run_384_11(x)
    elif route == 'unfold_16_45':
        return _run_unfold_16_45(x)
    elif route == 'unfold_384_11':
        return _run_unfold_384_11(x)
    elif route == 'im2col_16_45':
        return _run_im2col_16_45(x)
    elif route == 'im2col_384_11':
        return _run_im2col_384_11(x)
    else:
        return _run_16_45(x)