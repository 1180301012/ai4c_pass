import torch
import triton
import triton.language as tl

def pattern(in_2, in_1, in_4):
    # Match the slice + negate + concatenate pattern
    tmp_1 = in_2[Ellipsis, slice(None, 128, None)]
    tmp_2 = in_2[Ellipsis, slice(128, None, None)]
    tmp_3 = -tmp_2
    tmp_4 = torch.cat((tmp_3, tmp_1), dim=-1)
    
    # Also match the multiplication chain
    tmp_0 = in_2 * in_1
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    
    # Only return what's observable outside the matched subgraph
    return tmp_6

def replacement_args(in_2, in_1, in_4):
    return (in_2, in_1, in_4)

@triton.jit
def slice_negate_concat_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input tensor
    in_vals = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    
    # Apply slice+negate+concat logic: 
    # For positions 0-127: take values from 128-255 and negate them
    # For positions 128-255: take values from 0-127 unchanged
    # This effectively creates the pattern: [second_half_negated, first_half]
    
    # Simple element-wise transformation
    element_pos = offsets % 256
    
    # If in first half of result (0-127), take from second half of input and negate
    # If in second half of result (128-255), take from first half of input unchanged
    result = tl.where(element_pos < 128, 
                     -tl.load(in_ptr + (offsets + 128), mask=mask, other=0.0),
                     tl.load(in_ptr + (offsets - 128), mask=mask, other=0.0))
    
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def optimized_slice_negate_concat(in_2, in_1, in_4):
    N = in_2.numel()
    BLOCK_SIZE = 1024
    num_programs = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor for the slice+negate+concat pattern  
    tmp_4 = torch.empty_like(in_2)
    
    # Apply the optimized Triton kernel
    slice_negate_concat_kernel[(num_programs, 1, 1)](
        in_ptr=in_2,
        out_ptr=tmp_4,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    # Complete the computation chain
    tmp_0 = in_2 * in_1
    tmp_5 = tmp_4 * in_4
    tmp_6 = tmp_0 + tmp_5
    
    return tmp_6

def replacement_func():
    return optimized_slice_negate_concat