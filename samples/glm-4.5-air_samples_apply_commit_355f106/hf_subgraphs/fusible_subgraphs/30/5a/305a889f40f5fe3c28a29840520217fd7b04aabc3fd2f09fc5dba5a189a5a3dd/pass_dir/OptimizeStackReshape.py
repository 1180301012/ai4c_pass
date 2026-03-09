import torch
import triton
import triton.language as tl
import math

def pattern(in_1, in_3, target_shape):
    tmp_1 = -in_1
    tmp_2 = in_3[Ellipsis, slice(None, None, 2)]
    tmp_3 = torch.stack([tmp_1, tmp_2], -1)
    tmp_4 = tmp_3.reshape(target_shape)
    return tmp_4

def replacement_args(in_1, in_3, target_shape):
    return (in_1, in_3, target_shape)

@triton.jit
def optimized_stack_reshape_kernel(
    neg_in_1_ptr,
    sliced_in_3_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the two input tensors
    neg_in_1 = tl.load(neg_in_1_ptr + offsets, mask=mask, other=0.0)
    sliced_in_3 = tl.load(sliced_in_3_ptr + offsets, mask=mask, other=0.0)
    
    # Interleave the results: create equivalent to stack([...], -1) followed by reshape
    # For even offsets: take from neg_in_1, for odd offsets: take from sliced_in_3
    interleaved = tl.where(offsets % 2 == 0, neg_in_1, sliced_in_3)
    
    # Store the result
    tl.store(out_ptr + offsets, interleaved, mask=mask)

@torch.fx.wrap
def optimized_stack_reshape_wrapper(in_1, in_3, target_shape):
    # Create negative of in_1
    neg_in_1 = -in_1
    
    # Create slice of in_3
    sliced_in_3 = in_3[Ellipsis, slice(None, None, 2)]
    
    # Verify shapes are compatible
    assert neg_in_1.shape == sliced_in_3.shape, "Inputs must have compatible shapes for stacking"
    
    # Create output tensor
    N = neg_in_1.numel() * 2  # Stack doubles the size in the last dimension
    out = torch.empty(target_shape, device=in_1.device, dtype=in_1.dtype)
    
    # Calculate optimal grid size
    BLOCK_SIZE = 1024
    num_programs = math.ceil(N / BLOCK_SIZE)
    
    # Launch kernel
    optimized_stack_reshape_kernel[(num_programs,)](
        neg_in_1_ptr=neg_in_1,
        sliced_in_3_ptr=sliced_in_3,
        out_ptr=out,
        n_elements=N,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_stack_reshape_wrapper