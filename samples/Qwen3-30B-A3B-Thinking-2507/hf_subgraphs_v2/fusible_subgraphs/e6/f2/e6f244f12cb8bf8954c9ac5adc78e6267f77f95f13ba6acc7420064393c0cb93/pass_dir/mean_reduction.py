import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(
    x_ptr,
    out_ptr,
    batch_size,
    channels,
    height,
    width,
    BLOCK_H: tl.constexpr = 8,
    BLOCK_W: tl.constexpr = 8,
):
    b_idx = tl.program_id(0)
    c_idx = tl.program_id(1)
    start_idx = b_idx * channels * height * width + c_idx * height * width
    num_blocks_h = tl.cdiv(height, BLOCK_H)
    num_blocks_w = tl.cdiv(width, BLOCK_W)
    total = tl.zeros((1,), dtype=tl.float32)
    for h_block in range(num_blocks_h):
        for w_block in range(num_blocks_w):
            h_start = h_block * BLOCK_H
            w_start = w_block * BLOCK_W
            h_offset = tl.arange(0, BLOCK_H)[:, None] + h_start
            w_offset = tl.arange(0, BLOCK_W)[None, :] + w_start
            mask = (h_offset < height) & (w_offset < width)
            x_offset = start_idx + h_offset * width + w_offset
            x_tile = tl.load(x_ptr + x_offset, mask=mask, other=0.0)
            tile_sum = tl.sum(x_tile, axis=(0, 1))
            total += tile_sum
    mean_val = total / (height * width)
    tl.store(out_ptr + b_idx * channels + c_idx, mean_val)

@torch.fx.wrap
def triton_mean(x):
    B, C, H, W = x.shape
    out = torch.empty((B, C), dtype=x.dtype, device=x.device)
    grid = (B, C)
    mean_kernel[grid](x, out, B, C, H, W, BLOCK_H=8, BLOCK_W=8)
    return out.view(B, C, 1, 1)

def pattern(x):
    y = x.mean((2, 3), keepdim=True)
    return (x, y)

def replacement_args(x):
    return (x,)

def replacement_func():
    return triton_mean