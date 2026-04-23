import torch
import triton
import triton.language as tl

@triton.jit
def interpolate_kernel(
    in_ptr,
    out_ptr,
    batch_size,
    in_h,
    in_w,
    new_h,
    new_w,
    C,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr
):
    block_x = tl.program_id(0)
    block_y = tl.program_id(1)
    batch_ch = tl.program_id(2)
    batch = batch_ch // C
    c = batch_ch % C

    block_start_h = block_x * BLOCK_H
    block_start_w = block_y * BLOCK_W

    for h in range(BLOCK_H):
        for w in range(BLOCK_W):
            out_h = block_start_h + h
            out_w = block_start_w + w
            if out_h >= new_h or out_w >= new_w:
                continue
            in_h_coord = out_h // 2
            in_w_coord = out_w // 2
            in_idx = batch * C * in_h * in_w + c * in_h * in_w + in_h_coord * in_w + in_w_coord
            out_idx = batch * C * new_h * new_w + c * new_h * new_w + out_h * new_w + out_w
            val = tl.load(in_ptr + in_idx)
            tl.store(out_ptr + out_idx, val)

@torch.fx.wrap
def optimized_interpolate(y):
    batch = y.shape[0]
    C = y.shape[1]
    in_h = y.shape[2]
    in_w = y.shape[3]
    new_h = 40
    new_w = 40
    out = torch.empty_like(y)
    BLOCK_H = 16
    BLOCK_W = 16
    num_blocks_x = (new_h + BLOCK_H - 1) // BLOCK_H
    num_blocks_y = (new_w + BLOCK_W - 1) // BLOCK_W
    num_blocks_z = batch * C
    interpolate_kernel[(num_blocks_x, num_blocks_y, num_blocks_z)](
        y, out,
        batch, in_h, in_w, new_h, new_w, C,
        BLOCK_H, BLOCK_W
    )
    return out

def pattern(y):
    return torch.nn.functional.interpolate(y, size=(40,40), mode='nearest')

def replacement_args(y):
    return (y,)

def replacement_func():
    return optimized_interpolate