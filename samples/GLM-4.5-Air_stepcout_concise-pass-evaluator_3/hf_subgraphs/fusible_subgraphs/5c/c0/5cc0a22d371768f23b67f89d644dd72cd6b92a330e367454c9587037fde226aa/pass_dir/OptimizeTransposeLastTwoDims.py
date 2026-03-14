import torch
import triton
import triton.language as tl

def pattern(input_tensor):
    """Pattern: Transpose last two dimensions"""
    return input_tensor.transpose(-2, -1)

def replacement_args(input_tensor):
    return (input_tensor,)

@triton.jit
def optimized_transpose_kernel(
    input_ptr,
    output_ptr,
    input_stride_n, input_stride_c, input_stride_h, input_stride_w,
    output_stride_n, output_stride_c, output_stride_h, output_stride_w,
    n: tl.constexpr,
    c: tl.constexpr,
    h: tl.constexpr,
    w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized transpose kernel for last two dimensions"""
    pid = tl.program_id(0)
    block_size = BLOCK_SIZE
    
    # Process multiple elements per program
    offset = pid * block_size + tl.arange(0, BLOCK_SIZE)
    mask = offset < (n * c * h * w)
        
    # Convert linear offset to 4D coordinates for output (N,C,H,W)
    idx = offset
    w_idx = idx % w
    idx = idx // w
    h_idx = idx % h
    idx = idx // h
    c_idx = idx % c
    idx = idx // c
    n_idx = idx
    
    # For transposing H<->W dimensions, map output coordinates to input coordinates
    input_w_idx = h_idx  # W in output comes from H in input
    input_h_idx = w_idx  # H in output comes from W in input
    input_c_idx = c_idx
    input_n_idx = n_idx
    
    # Calculate input offset
    input_offset = (
        input_n_idx * input_stride_n +
        input_c_idx * input_stride_c +
        input_h_idx * input_stride_h +
        input_w_idx * input_stride_w
    )
    
    # Calculate output offset
    output_offset = (
        n_idx * output_stride_n +
        c_idx * output_stride_c +
        h_idx * output_stride_h +
        w_idx * output_stride_w
    )
    
    # Load from input and store to output
    input_val = tl.load(input_ptr + input_offset, mask=mask)
    tl.store(output_ptr + output_offset, input_val, mask=mask)

@torch.fx.wrap
def optimized_transpose_wrapper(input_tensor):
    """Wrapper for optimized last two dimensions transpose"""
    # Input tensor shape: [n, c, h, w] = [1, 16, 196, 48]
    # Output tensor shape: [n, c, w, h] = [1, 16, 48, 196]
    
    n, c, h, w = input_tensor.shape
    
    # Create output tensor with last two dimensions transposed
    output = torch.empty((n, c, w, h), dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate block size for good GPU occupancy
    total_elements = n * c * h * w
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Calculate strides (assuming contiguous tensors)
    input_stride_n = input_tensor.stride(0)
    input_stride_c = input_tensor.stride(1)
    input_stride_h = input_tensor.stride(2)
    input_stride_w = input_tensor.stride(3)
    
    output_stride_n = output.stride(0)
    output_stride_c = output.stride(1)
    output_stride_h = output.stride(2)
    output_stride_w = output.stride(3)
    
    # Launch kernel
    optimized_transpose_kernel[(num_programs,)](
        input_ptr=input_tensor,
        output_ptr=output,
        input_stride_n=input_stride_n,
        input_stride_c=input_stride_c,
        input_stride_h=input_stride_h,
        input_stride_w=input_stride_w,
        output_stride_n=output_stride_n,
        output_stride_c=output_stride_c,
        output_stride_h=output_stride_h,
        output_stride_w=output_stride_w,
        n=n,
        c=c,
        h=h,
        w=w,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_transpose_wrapper