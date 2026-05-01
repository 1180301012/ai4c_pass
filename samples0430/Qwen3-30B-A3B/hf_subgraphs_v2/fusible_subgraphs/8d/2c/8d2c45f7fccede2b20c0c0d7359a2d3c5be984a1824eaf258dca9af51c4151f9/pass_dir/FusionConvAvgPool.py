import torch
import triton
import triton.language as tl

# Pattern matching function
def pattern(in_0, in_1):
    conv = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    pool = torch.nn.functional.avg_pool2d(conv, 2, 2, 0, False, True, None)
    return pool

# Argument extraction function
def replacement_args(in_1, in_0):
    return (in_1, in_0)

# Fused kernel
@triton.jit
def fused_conv_avg_pool_kernel(
    input_ptr,
    weights_ptr,
    output_ptr,
    batch_size,
    c_in,
    c_out,
    h,
    w,
    h_out,
    w_out,
    BLOCK_SIZE: tl.constexpr,
):
    idx = tl.program_id(0)
    batch_idx = idx // (c_out * h_out * w_out)
    remainder = idx % (c_out * h_out * w_out)
    c_out_idx = remainder // (h_out * w_out)
    i = (remainder // w_out) % h_out
    j = remainder % w_out

    output_idx = batch_idx * c_out * h_out * w_out + c_out_idx * h_out * w_out + i * w_out + j

    sum_val = tl.zeros((1,), dtype=tl.float32)

    # Process each of the 4 spatial positions (2x2 block)
    sp_i = 2 * i
    sp_j = 2 * j
    input_offset = batch_idx * c_in * h * w + sp_i * c_in * w + sp_j * c_in
    input_data0 = tl.load(input_ptr + input_offset, mask=tl.arange(0, c_in) < c_in, other=0.0)
    weight_offset = c_out_idx * c_in
    weight_data = tl.load(weights_ptr + weight_offset, mask=tl.arange(0, c_in) < c_in, other=0.0)
    dot0 = tl.dot(input_data0, weight_data)
    sum_val += dot0

    sp_i = 2 * i + 1
    sp_j = 2 * j
    input_offset = batch_idx * c_in * h * w + sp_i * c_in * w + sp_j * c_in
    input_data1 = tl.load(input_ptr + input_offset, mask=tl.arange(0, c_in) < c_in, other=0.0)
    dot1 = tl.dot(input_data1, weight_data)
    sum_val += dot1

    sp_i = 2 * i
    sp_j = 2 * j + 1
    input_offset = batch_idx * c_in * h * w + sp_i * c_in * w + sp_j * c_in
    input_data2 = tl.load(input_ptr + input_offset, mask=tl.arange(0, c_in) < c_in, other=0.0)
    dot2 = tl.dot(input_data2, weight_data)
    sum_val += dot2

    sp_i = 2 * i + 1
    sp_j = 2 * j + 1
    input_offset = batch_idx * c_in * h * w + sp_i * c_in * w + sp_j * c_in
    input_data3 = tl.load(input_ptr + input_offset, mask=tl.arange(0, c_in) < c_in, other=0.0)
    dot3 = tl.dot(input_data3, weight_data)
    sum_val += dot3

    sum_val *= 0.25
    tl.store(output_ptr + output_idx, sum_val)

@torch.fx.wrap
def fused_conv_avg_pool(x, weights):
    batch_size, c_in, h, w = x.shape
    c_out = weights.shape[0]
    h_out = h // 2
    w_out = w // 2

    output = torch.empty((batch_size, c_out, h_out, w_out), dtype=x.dtype, device=x.device)

    num_elements = batch_size * c_out * h_out * w_out
    grid = (num_elements,)

    fused_conv_avg_pool_kernel[grid](
        x, weights, output,
        batch_size, c_in, c_out, h, w, h_out, w_out,
        BLOCK_SIZE=1
    )

    return output

def replacement_func():
    return fused_conv_avg_pool