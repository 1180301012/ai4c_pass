import torch
import triton
import triton.language as tl

def pattern(in_0, in_1, in_2, in_3):
    conv = torch.conv2d(in_3, in_1, in_0, (1, 1), (0, 0), (1, 1), 1)
    dropout = torch.nn.functional.dropout(conv, 0.0, False, False)
    add = dropout + in_2
    return add

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

@triton.jit
def conv2d_add_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    hidden_ptr,
    output_ptr,
    batch_size,
    in_ch,
    out_ch,
    h,
    w,
    BLOCK_SIZE: tl.constexpr
):
    block_start = tl.program_id(0) * BLOCK_SIZE
    start_idx = block_start + tl.arange(0, BLOCK_SIZE)
    mask = start_idx < out_ch * h * w
    c_out = start_idx // (h * w)
    remainder = start_idx % (h * w)
    h_idx = remainder // w
    w_idx = remainder % w

    hidden_val = tl.load(hidden_ptr + c_out * (h * w) + h_idx * w + w_idx, mask=mask)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for c_in in range(in_ch):
        input_val = tl.load(input_ptr + c_in * (h * w) + h_idx * w + w_idx, mask=mask)
        weight_val = tl.load(weight_ptr + c_out * in_ch + c_in, mask=mask)
        acc += input_val * weight_val

    bias_val = tl.load(bias_ptr + c_out, mask=mask)
    acc += bias_val
    acc += hidden_val

    tl.store(output_ptr + start_idx, acc, mask=mask)

@torch.fx.wrap
def optimized_conv_add(in_0, in_1, in_2, in_3):
    weight_2d = None

# Removed view() call to avoid unauthorized operation; kernel handles 4D weight directly
    batch_size = in_3.size(0)
    in_ch = in_3.size(1)
    h = in_3.size(2)
    w = in_3.size(3)
    out_ch = in_1.size(0)

    output = torch.empty_like(in_2)
    BLOCK_SIZE = 128
    num_blocks = (out_ch * h * w + BLOCK_SIZE - 1) // BLOCK_SIZE

    conv2d_add_kernel[(num_blocks,)](
        in_3,
        in_1,
        in_0,
        in_2,
        output,
        batch_size,
        in_ch,
        out_ch,
        h,
        w,
        BLOCK_SIZE
    )

    return output

def replacement_func():
    return optimized_conv_add