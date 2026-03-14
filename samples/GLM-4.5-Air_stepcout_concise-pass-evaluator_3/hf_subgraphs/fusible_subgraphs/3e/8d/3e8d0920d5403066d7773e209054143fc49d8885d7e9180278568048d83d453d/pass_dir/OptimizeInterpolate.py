import torch
import triton
import triton.language as tl
import math

def pattern(tmp_5):
    # Interpolate operation - this is the key pattern to match from the graphs
    tmp_6 = torch.nn.functional.interpolate(tmp_5, (256, 256), None, 'bilinear', False)
    return tmp_6

def replacement_args(tmp_5):
    return (tmp_5,)

@triton.jit
def optimized_interpolate_kernel(
    input_ptr,
    output_ptr,
    input_n,
    input_c,
    input_h,
    input_w,
    target_h,
    target_w,
    BLOCK_SIZE_X: tl.constexpr,
    BLOCK_SIZE_Y: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Output coordinates
    out_h_idx = pid // target_w
    out_w_idx = pid % target_w
    
    # Skip if out of bounds
    if out_h_idx >= target_h or out_w_idx >= target_w:
        return
    
    # Get batch and channel from thread IDs
    batch_idx = tl.program_id(1)
    channel_idx = tl.program_id(2)
    
    # Skip if out of bounds
    if batch_idx >= input_n or channel_idx >= input_c:
        return
    
    # Input coordinates (bilinear interpolation)
    in_h_float = out_h_idx * (input_h - 1) / max(1, target_h - 1)
    in_w_float = out_w_idx * (input_w - 1) / max(1, target_w - 1)
    
    in_h0 = int(in_h_float)
    in_w0 = int(in_w_float)
    in_h1 = min(in_h0 + 1, input_h - 1)
    in_w1 = min(in_w0 + 1, input_w - 1)
    
    # Boundary checks
    in_h0 = tl.math.max(0, tl.math.min(in_h0, input_h - 1))
    in_w0 = tl.math.max(0, tl.math.min(in_w0, input_w - 1))
    in_h1 = tl.math.max(0, tl.math.min(in_h1, input_h - 1))
    in_w1 = tl.math.max(0, tl.math.min(in_w1, input_w - 1))
    
    # Weights for bilinear interpolation
    w_h = in_h_float - in_h0
    w_w = in_w_float - in_w0
    
    # Load 4 corner pixels
    idx00 = batch_idx * input_c * input_h * input_w + channel_idx * input_h * input_w + in_h0 * input_w + in_w0
    idx01 = batch_idx * input_c * input_h * input_w + channel_idx * input_h * input_w + in_h0 * input_w + in_w1
    idx10 = batch_idx * input_c * input_h * input_w + channel_idx * input_h * input_w + in_h1 * input_w + in_w0
    idx11 = batch_idx * input_c * input_h * input_w + channel_idx * input_h * input_w + in_h1 * input_w + in_w1
    
    v00 = tl.load(input_ptr + idx00)
    v01 = tl.load(input_ptr + idx01)
    v10 = tl.load(input_ptr + idx10)
    v11 = tl.load(input_ptr + idx11)
    
    # Bilinear interpolation
    v0 = v00 * (1 - w_w) + v01 * w_w
    v1 = v10 * (1 - w_w) + v11 * w_w
    out_val = v0 * (1 - w_h) + v1 * w_h
    
    # Store result
    out_idx = batch_idx * input_c * target_h * target_w + channel_idx * target_h * target_w + out_h_idx * target_w + out_w_idx
    tl.store(output_ptr + out_idx, out_val)

@torch.fx.wrap
def optimized_interpolate(tmp_5, target_size=(256, 256)):
    input_n, input_c, input_h, input_w = tmp_5.shape
    target_h, target_w = target_size
    
    output = torch.empty((input_n, input_c, target_h, target_w), dtype=tmp_5.dtype, device=tmp_5.device)
    
    BLOCK_SIZE_X = 16
    BLOCK_SIZE_Y = 16
    
    # Calculate grid size
    total_elements = input_n * input_c * target_h * target_w
    grid_size_x = (total_elements + 255) // 256
    
    # Launch the optimized interpolate kernel
    optimized_interpolate_kernel[(
        grid_size_x,
        input_n,
        input_c
    )](
        tmp_5,
        output,
        input_n,
        input_c,
        input_h,
        input_w,
        target_h,
        target_w,
        BLOCK_SIZE_X,
        BLOCK_SIZE_Y,
    )
    
    return output

def replacement_func():
    return optimized_interpolate