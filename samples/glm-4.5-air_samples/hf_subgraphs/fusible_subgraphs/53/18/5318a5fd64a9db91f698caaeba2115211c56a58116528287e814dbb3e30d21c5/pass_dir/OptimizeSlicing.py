import torch
import triton
import triton.language as tl

# Pattern matching function for slicing operation
def pattern(in_0, in_1, in_2):
    # Just match the slicing operations that produce observable outputs
    tmp_1 = in_1[slice(None, None, None), slice(None, None, None), slice(1, None, None), slice(None, None, None)]
    # The second slice on in_2 is not observable in the final return
    return tmp_1

# Optimized slicing kernel
@triton.jit
def slice_kernel(
    in_ptr,
    out_ptr,
    batch,
    heads,
    seq_len,
    dim,
    start_idx,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total_elements = batch * heads * (seq_len - start_idx) * dim
    n_elements = total_elements
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Convert linear offset to tensor coordinates
    offset_idx = offsets
    dim_idx = offset_idx % dim
    offset_idx = offset_idx // dim
    seq_idx = offset_idx % (seq_len - start_idx) + start_idx
    offset_idx = offset_idx // (seq_len - start_idx)
    head_idx = offset_idx % heads
    batch_idx = offset_idx // heads
    
    # Calculate original pointer offset
    original_offset = (batch_idx * heads * seq_len * dim + 
                      head_idx * seq_len * dim + 
                      seq_idx * dim + 
                      dim_idx)
    
    # Load from input and store to output
    val = tl.load(in_ptr + original_offset, other=0.0, mask=(seq_idx < seq_len) & (dim_idx < dim))
    tl.store(out_ptr + offsets, val, mask=mask)

# Kernel wrapper for slicing
@torch.fx.wrap
def optimized_slice(input_tensor, start_idx=1):
    batch, heads, seq_len, dim = input_tensor.shape
    
    # Output shape after slicing
    output_shape = (batch, heads, seq_len - start_idx, dim)
    output = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024
    total_elements = batch * heads * (seq_len - start_idx) * dim
    grid_size = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    slice_kernel[grid_size](
        input_tensor,
        output,
        batch,
        heads,
        seq_len,
        dim,
        start_idx,
        BLOCK_SIZE
    )
    
    return output

# Replacement function
def replacement_func():
    def optimized_slicing(in_0, in_1, in_2):
        # Just optimize the slicing operation on in_1
        tmp_1 = optimized_slice(in_1, start_idx=1)
        
        # Return the sliced result (pattern only matches one return)
        return tmp_1
    
    return optimized_slicing

# Argument extraction function
def replacement_args(in_0, in_1, in_2):
    return (in_0, in_1, in_2)