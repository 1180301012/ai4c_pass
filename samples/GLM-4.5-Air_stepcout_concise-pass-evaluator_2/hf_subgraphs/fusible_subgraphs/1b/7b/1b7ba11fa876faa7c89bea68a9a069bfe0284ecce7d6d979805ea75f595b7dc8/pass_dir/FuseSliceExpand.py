import torch
import triton
import triton.language as tl

def pattern(in_1, slice_dim1, expand_dim0):
    """
    Pattern: slice followed by expand operation
    Matches: tmp_2 = tmp_1[slice(None, None, None), slice(None, X, None)]
             tmp_3 = tmp_2.expand(Y, X)
    """
    # Slice along dimension 1
    tmp_2 = in_1[slice(None, None, None), slice(None, slice_dim1, None)]
    # Expand the sliced tensor
    tmp_3 = tmp_2.expand(expand_dim0, slice_dim1)
    return tmp_3

def replacement_args(in_1, slice_dim1, expand_dim0):
    return (in_1, slice_dim1, expand_dim0)

@triton.jit
def fused_slice_expand_kernel(
    input_ptr,
    output_ptr,
    input_stride_0,
    input_stride_1,
    n_elements,
    slice_dim1: tl.constexpr,
    expand_dim0: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused kernel for slice + expand operations"""
    pid = tl.program_id(0)
    
    # Calculate output indices
    row = pid // slice_dim1
    col = pid % slice_dim1
    
    if row >= expand_dim0 or col >= slice_dim1:
        return
        
    # Calculate input index (only read first row of input)
    input_row = 0
    input_col = col
    input_idx = input_row * input_stride_0 + input_col * input_stride_1
    
    # Load input value
    input_val = tl.load(input_ptr + input_idx, other=0)
    
    # Calculate output index  
    output_idx = row * slice_dim1 + col
    
    # Store output value
    tl.store(output_ptr + output_idx, input_val)

@torch.fx.wrap
def fused_slice_expand_op(in_1, slice_dim1, expand_dim0):
    """Optimized implementation of slice + expand fusion"""
    input_tensor = in_1
    output_shape = (expand_dim0, slice_dim1)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Get input strides
    input_stride_0 = input_tensor.stride(0)
    input_stride_1 = input_tensor.stride(1)
    
    # Calculate total elements in output
    n_elements = expand_dim0 * slice_dim1
    
    # Block size tuning
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    fused_slice_expand_kernel[grid](
        input_tensor,
        output,
        input_stride_0,
        input_stride_1,
        n_elements,
        slice_dim1,
        expand_dim0,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_slice_expand_op