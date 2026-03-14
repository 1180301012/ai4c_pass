import torch
import triton
import triton.language as tl

def pattern(in_0):
    """
    Pattern: slice operation with None dimension insertion
    Matches: tmp_4 = tmp_0[slice(None, None, None), None, None, slice(None, None, None)]
    This pattern takes [batch, seq_len] and converts to [batch, 1, 1, seq_len]
    """
    # Slice and add None dimensions
    tmp_4 = in_0[slice(None, None, None), None, None, slice(None, None, None)]
    return tmp_4

def replacement_args(in_0):
    return (in_0,)

@triton.jit
def optimized_slice_none_dims_kernel(
    input_ptr,
    output_ptr,
    input_batch_size,
    input_seq_len,
    output_batch_size,
    output_seq_len,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for slice + None dimension operations"""
    pid = tl.program_id(0)
    
    # Calculate which element we're processing
    if pid >= output_batch_size * output_seq_len:
        return
    
    # Calculate batch and sequence indices
    batch_idx = pid // output_seq_len
    seq_idx = pid % output_seq_len
    
    if batch_idx >= input_batch_size:
        return
    
    # Input index calculation (no None dims in input)
    input_idx = batch_idx * input_seq_len + seq_idx
    
    # Output index calculation (with None dims)  
    # Shape is [batch, 1, 1, seq] -> offset = batch * (1 * 1 * seq) + seq
    output_idx = batch_idx * (1 * 1 * output_seq_len) + seq_idx
    
    # Load input value and store to output
    input_val = tl.load(input_ptr + input_idx, other=0)
    tl.store(output_ptr + output_idx, input_val)

@torch.fx.wrap  
def optimized_slice_none_dims_op(in_0):
    """Optimized implementation of slice + None dimensions"""
    input_tensor = in_0
    input_shape = input_tensor.shape
    
    # The pattern assumes input is [batch, seq_len] and output is [batch, 1, 1, seq_len]
    input_batch_size, input_seq_len = input_shape
    output_batch_size, output_seq_len = input_batch_size, input_seq_len
    
    # Create output tensor with new shape
    output_shape = (output_batch_size, 1, 1, output_seq_len)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    # Calculate total elements in output
    total_elements = output_batch_size * output_seq_len
    
    # Block size tuning
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    optimized_slice_none_dims_kernel[grid](
        input_tensor,
        output,
        input_batch_size,
        input_seq_len,
        output_batch_size,
        output_seq_len,
        BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return optimized_slice_none_dims_op