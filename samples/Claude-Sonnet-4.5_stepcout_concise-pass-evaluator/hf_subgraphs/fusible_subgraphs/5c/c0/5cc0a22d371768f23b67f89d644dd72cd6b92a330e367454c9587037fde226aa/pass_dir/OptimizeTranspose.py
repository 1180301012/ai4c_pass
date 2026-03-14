import torch
import triton
import triton.language as tl

def pattern(in_2):
    """Pattern to match: transpose(-2, -1)"""
    tmp_4 = in_2.transpose(-2, -1)
    return tmp_4

def replacement_args(in_2):
    return (in_2,)

@triton.jit
def transpose_kernel(
    input_ptr, output_ptr,
    batch, dim0, dim1, dim2,
    stride_b, stride_d0, stride_d1, stride_d2,
    out_stride_b, out_stride_d0, out_stride_d1, out_stride_d2,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    total_elements = batch * dim0 * dim1 * dim2
    mask = offsets < total_elements
    
    # Compute input indices
    idx = offsets
    b = idx // (dim0 * dim1 * dim2)
    remainder = idx % (dim0 * dim1 * dim2)
    d0 = remainder // (dim1 * dim2)
    remainder = remainder % (dim1 * dim2)
    d1 = remainder // dim2
    d2 = remainder % dim2
    
    # Load from input
    input_offset = b * stride_b + d0 * stride_d0 + d1 * stride_d1 + d2 * stride_d2
    data = tl.load(input_ptr + input_offset, mask=mask, other=0.0)
    
    # For transpose(-2, -1), we swap dim1 and dim2
    # So output indices are: [b, d0, d2, d1]
    output_offset = b * out_stride_b + d0 * out_stride_d0 + d2 * out_stride_d1 + d1 * out_stride_d2
    tl.store(output_ptr + output_offset, data, mask=mask)

@torch.fx.wrap
def optimized_transpose(input):
    # Input shape: [batch, dim0, dim1, dim2]
    # Output shape: [batch, dim0, dim2, dim1] (swap last two dims)
    batch, dim0, dim1, dim2 = input.shape
    output = torch.empty((batch, dim0, dim2, dim1), device=input.device, dtype=input.dtype)
    
    total_elements = batch * dim0 * dim1 * dim2
    BLOCK_SIZE = 1024
    num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Compute strides
    stride_b = dim0 * dim1 * dim2
    stride_d0 = dim1 * dim2
    stride_d1 = dim2
    stride_d2 = 1
    
    out_stride_b = dim0 * dim2 * dim1
    out_stride_d0 = dim2 * dim1
    out_stride_d1 = dim1
    out_stride_d2 = 1
    
    transpose_kernel[(num_programs,)](
        input, output,
        batch, dim0, dim1, dim2,
        stride_b, stride_d0, stride_d1, stride_d2,
        out_stride_b, out_stride_d0, out_stride_d1, out_stride_d2,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return optimized_transpose