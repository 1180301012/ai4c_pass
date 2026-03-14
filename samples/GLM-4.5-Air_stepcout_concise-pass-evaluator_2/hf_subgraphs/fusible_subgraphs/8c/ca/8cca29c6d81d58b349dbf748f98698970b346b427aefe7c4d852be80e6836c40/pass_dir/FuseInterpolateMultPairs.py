import torch
import triton
import triton.language as tl
import math

def pattern(in_0, in_1, in_2, in_3):
    """
    Pattern matching for two independent interpolate + multiply operations.
    Each pair: interpolate(input, size, mode='nearest') * multiplier
    """
    # First pair: interpolate in_0 * in_2
    tmp_0 = torch.nn.functional.interpolate(in_0, size=(64, 48), mode='nearest')
    tmp_1 = in_2 * tmp_0
    
    # Second pair: interpolate in_1 * in_3
    tmp_2 = torch.nn.functional.interpolate(in_1, size=(32, 24), mode='nearest')
    tmp_3 = in_3 * tmp_2
    
    return (tmp_1, tmp_3)

def replacement_args(in_0, in_1, in_2, in_3):
    return (in_0, in_1, in_2, in_3)

# Optimized kernel for interpolate + multiply fusion
@triton.jit
def interpolate_mult_kernel_pair(
    input_ptr,
    output_ptr,
    multiplier_ptr,
    batch_size,
    input_channels,
    interp_size_h,
    interp_size_w,
    input_size_h,
    input_size_w,
    stride_h,
    stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one output element
    pid = tl.program_id(0)
    
    # Calculate output coordinates
    h = pid // interp_size_w
    w = pid % interp_size_w
    c = tl.program_id(1)
    b = tl.program_id(2)
    
    if h >= interp_size_h or w >= interp_size_w or c >= input_channels or b >= batch_size:
        return
    
    # Calculate corresponding input coordinates for nearest neighbor interpolation
    src_h = h // stride_h
    src_w = w // stride_w
    src_h = tl.max(0, tl.min(src_h, input_size_h - 1))
    src_w = tl.max(0, tl.min(src_w, input_size_w - 1))
    
    # Calculate linear indices
    input_idx = b * input_channels * input_size_h * input_size_w + \
                c * input_size_h * input_size_w + \
                src_h * input_size_w + src_w
    
    output_idx = b * input_channels * interp_size_h * interp_size_w + \
                 c * interp_size_h * interp_size_w + \
                 h * interp_size_w + w
    
    # Load input and multiplier, multiply, store result
    input_val = tl.load(input_ptr + input_idx, other=0.0)
    multiplier_val = tl.load(multiplier_ptr + output_idx, other=0.0)
    result = input_val * multiplier_val
    tl.store(output_ptr + output_idx, result)

@torch.fx.wrap
def fused_interpolate_mult_pair(input_tensor, multiplier_tensor, input_shape, output_shape):
    """
    Fused interpolation + multiplication for one pair
    """
    batch_size, input_channels, input_size_h, input_size_w = input_shape
    interp_size_h, interp_size_w = output_shape
    
    # Calculate stride for nearest neighbor interpolation
    stride_h = interp_size_h // input_size_h if input_size_h > 0 else 1
    stride_w = interp_size_w // input_size_w if input_size_w > 0 else 1
    
    output = torch.empty((batch_size, input_channels, interp_size_h, interp_size_w), 
                        dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate grid size
    total_elements = batch_size * input_channels * interp_size_h * interp_size_w
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    interpolate_mult_kernel_pair[
        (num_programs, input_channels, batch_size),
        (BLOCK_SIZE,)
    ](
        input_ptr=input_tensor,
        output_ptr=output,
        multiplier_ptr=multiplier_tensor,
        batch_size=batch_size,
        input_channels=input_channels,
        interp_size_h=interp_size_h,
        interp_size_w=interp_size_w,
        input_size_h=input_size_h,
        input_size_w=input_size_w,
        stride_h=stride_h,
        stride_w=stride_w,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

@torch.fx.wrap
def optimized_forward(in_0, in_1, in_2, in_3):
    """
    Optimized forward with two fused interpolate-multiply pairs computed in parallel
    """
    # Get input shapes
    shape_0 = in_0.shape
    shape_1 = in_1.shape
    
    # First pair: interpolate in_0 to (64, 48) and multiply by in_2
    result_1 = fused_interpolate_mult_pair(in_0, in_2, shape_0, (64, 48))
    
    # Second pair: interpolate in_1 to (32, 24) and multiply by in_3  
    result_2 = fused_interpolate_mult_pair(in_1, in_3, shape_1, (32, 24))
    
    return (result_1, result_2)

def replacement_func():
    return optimized_forward