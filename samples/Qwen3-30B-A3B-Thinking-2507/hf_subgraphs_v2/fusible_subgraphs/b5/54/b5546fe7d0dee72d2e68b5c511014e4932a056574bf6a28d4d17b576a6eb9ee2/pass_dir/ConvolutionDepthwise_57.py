import torch
import triton
import triton.language as tl

def pattern(in_5, in_1, in_0):
    return torch.conv2d(in_5, in_1, in_0, (1, 1), (3, 3), (1, 1), 57)

def replacement_args(in_5, in_1, in_0):
    return (in_5, in_1, in_0)

@triton.jit
def depthwise_conv2d_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    H, W, n,
    BLOCK_H: tl.constexpr,
    BLOCK_W: tl.constexpr,
    BLOCK_C: tl.constexpr
):
    out_h_start = tl.program_id(0) * BLOCK_H
    out_w_start = tl.program_id(1) * BLOCK_W
    out_c_start = tl.program_id(2) * BLOCK_C
    out_c = out_c_start + tl.thread_id(2)
    if out_c >= n:
        return

    acc = tl.zeros((1,), dtype=tl.float32)
    for kh in range(7):
        for kw in range(7):
            in_h = out_h_start + tl.thread_id(0) + kh - 3
            in_w = out_w_start + tl.thread_id(1) + kw - 3
            if in_h < 0 or in_h >= H or in_w < 0 or in_w >= W:
                val = 0.0
            else:
                input_index = 0 * (n * H * W) + out_c * (H * W) + in_h * W + in_w
                val = tl.load(input_ptr + input_index)
            weight_index = out_c * (1 * 7 * 7) + 0 * (7 * 7) + kh * 7 + kw
            w_val = tl.load(weight_ptr + weight_index)
            val_fp32 = val.to(tl.float32)
            w_fp32 = w_val.to(tl.float32)
            acc = acc + val_fp32 * w_fp32
    bias_val = tl.load(bias_ptr + out_c)
    bias_fp32 = bias_val.to(tl.float32)
    acc = acc + bias_fp32
    acc_fp16 = acc.to(tl.float16)
    output_index = 0 * (n * H * W) + out_c * (H * W) + (out_h_start + tl.thread_id(0)) * W + (out_w_start + tl.thread_id(1))
    tl.store(output_ptr + output_index, acc_fp16)

@torch.fx.wrap
def custom_conv2d(x, w, b):
    n = w.shape[0]
    H = x.shape[2]
    W = x.shape[3]
    out = torch.empty(1, n, H, W, dtype=x.dtype, device=x.device)
    BLOCK_H = 16
    BLOCK_W = 16
    BLOCK_C = 4
    grid_x = (H + BLOCK_H - 1) // BLOCK_H
    grid_y = (W + BLOCK_W - 1) // BLOCK_W
    grid_z = (n + BLOCK_C - 1) // BLOCK_C
    depthwise_conv2d_kernel[(grid_x, grid_y, grid_z)](
        x, w, b, out, H, W, n, BLOCK_H, BLOCK_W, BLOCK_C
    )
    return out

def replacement_func():
    return custom_conv2d