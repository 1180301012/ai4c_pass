import torch
import triton
import triton.language as tl

def pattern(in_0, in_1):
    conv = torch.conv2d(in_1, in_0, None, (1, 1), (0, 0), (1, 1), 1)
    unfold = torch.nn.functional.unfold(conv, kernel_size=(2, 2), stride=(2, 2))
    reshape = unfold.reshape(1, 128, 4, -1)
    return reshape

def replacement_args(in_0, in_1):
    return (in_0, in_1)

@triton.jit
def fused_conv_unfold_kernel(
    input_ptr,
    weight_ptr,
    output_ptr,
    in_channels,
    out_channels,
    in_h,
    in_w,
    out_h,
    out_w,
    stride,
):
    pid_h = tl.program_id(0)
    pid_w = tl.program_id(1)
    tid = tl.thread_id(0)
    
    k = tid
    
    if k >= out_channels:
        return
    
    row_start = pid_h * stride
    col_start = pid_w * stride
    
    positions = [
        (row_start, col_start),
        (row_start, col_start + 1),
        (row_start + 1, col_start),
        (row_start + 1, col_start + 1)
    ]
    
    block_id = pid_h * out_w + pid_w
    
    for pos_idx, (row, col) in enumerate(positions):
        dot = 0.0
        for c in range(in_channels):
            input_offset = c * (in_h * in_w) + row * in_w + col
            input_val = tl.load(input_ptr + input_offset)
            weight_val = tl.load(weight_ptr + k * in_channels + c)
            dot += input_val * weight_val
        output_offset = k * (4 * (out_h * out_w)) + pos_idx * (out_h * out_w) + block_id
        tl.store(output_ptr + output_offset, dot)

@torch.fx.wrap
def fused_conv_unfold(in_0, in_1):
    batch_size = 1
    in_channels = 256
    out_channels = 128
    in_h = 32
    in_w = 32
    stride = 2
    
    out_h = in_h // stride
    out_w = in_w // stride
    output_size = out_h * out_w
    
    output = torch.empty(1, out_channels, 4, output_size, dtype=in_1.dtype, device=in_1.device)
    
    input_contig = in_1.contiguous()
    weight_contig = in_0.contiguous()
    
    grid_h = out_h
    grid_w = out_w
    BLOCK_SIZE = 128
    
    fused_conv_unfold_kernel[(grid_h, grid_w), (BLOCK_SIZE)](
        input_contig,
        weight_contig,
        output,
        in_channels,
        out_channels,
        in_h,
        in_w,
        out_h,
        out_w,
        stride
    )
    
    return output

def replacement_func():
    return fused_conv_unfold