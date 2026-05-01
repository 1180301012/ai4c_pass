import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    linear = torch.nn.functional.linear(in_2, in_1, in_0)
    tmp_3 = torch.sigmoid(linear)
    tmp_4 = tmp_3.view(1, 64, 1, 1)
    tmp_5 = in_3 * tmp_4
    return tmp_5

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def fused_linear_sigmoid_multiply_kernel(
    in_2_ptr, in_1_ptr, in_0_ptr, in_3_ptr, out_ptr,
    B: tl.constexpr, C: tl.constexpr, H: tl.constexpr, W: tl.constexpr,
    BLOCK_B: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_H: tl.constexpr, BLOCK_W: tl.constexpr,
):
    b_start = tl.program_id(0) * BLOCK_B
    c_start = tl.program_id(1) * BLOCK_C
    b = tl.arange(0, BLOCK_B) + b_start
    c = tl.arange(0, BLOCK_C) + c_start
    b = tl.where(b < B, b, 0)
    c = tl.where(c < C, c, 0)
    
    in_2 = tl.load(in_2_ptr + b[:, None] * 8 + tl.arange(0, 8), mask=b[:, None] < B)
    in_1 = tl.load(in_1_ptr + c[:, None] * 8 + tl.arange(0, 8), mask=c[:, None] < C)
    bias = tl.load(in_0_ptr + c[:, None], mask=c[:, None] < C)
    
    linear = tl.zeros((BLOCK_B, BLOCK_C))
    for k in range(8):
        linear += in_2[:, k] * in_1[:, k]
    linear += bias[None, :]
    sigmoid_val = 1.0 / (1.0 + tl.exp(-linear))
    
    h = tl.arange(0, BLOCK_H) + tl.program_id(2) * BLOCK_H
    w = tl.arange(0, BLOCK_W)
    h = tl.where(h < H, h, 0)
    w = tl.where(w < W, w, 0)
    
    in_3_idx = b[:, None, None] * (C * H * W) + c[:, None, None] * (H * W) + h[None, :, None] * W + w[None, None, :]
    in_3_data = tl.load(in_3_ptr + in_3_idx, mask=(b[:, None, None] < B) & (c[:, None, None] < C) & (h[None, :, None] < H) & (w[None, None, :] < W))
    out_data = sigmoid_val[:, :, None, None] * in_3_data
    tl.store(out_ptr + in_3_idx, out_data, mask=(b[:, None, None] < B) & (c[:, None, None] < C) & (h[None, :, None] < H) & (w[None, None, :] < W))

@torch.fx.wrap
def fused_linear_sigmoid_multiply(in_0, in_1, in_2, in_3):
    B, _ = in_2.shape
    C = 64
    H, W = in_3.shape[2], in_3.shape[3]
    
    BLOCK_B, BLOCK_C = 4, 8
    BLOCK_H, BLOCK_W = 8, 8
    grid_x = (B + BLOCK_B - 1) // BLOCK_B
    grid_y = (C + BLOCK_C - 1) // BLOCK_C
    grid_z = (H + BLOCK_H - 1) // BLOCK_H
    
    out = torch.empty_like(in_3)
    fused_linear_sigmoid_multiply_kernel[(grid_x, grid_y, grid_z)](
        in_2, in_1, in_0, in_3, out,
        B, C, H, W,
        BLOCK_B, BLOCK_C, BLOCK_H, BLOCK_W
    )
    return out

def replacement_func():
    return fused_linear_sigmoid_multiply