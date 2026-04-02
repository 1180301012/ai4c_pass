import torch
import triton
import triton.language as tl

def pattern(tmp_0, in_1, in_2):
    # First just match the multiply-add operation
    tmp_1 = in_2 * in_1
    tmp_2 = tmp_1 + tmp_0
    return tmp_2

def replacement_args(tmp_0, in_1, in_2):
    return (tmp_0, in_1, in_2)

@triton.jit
def fused_multiply_add_kernel(
    in0_ptr,
    in1_ptr, 
    in2_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of data
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load inputs
    in0 = tl.load(in0_ptr + offsets, mask=mask, other=0.0)
    in1 = tl.load(in1_ptr + offsets, mask=mask, other=0.0) 
    in2 = tl.load(in2_ptr + offsets, mask=mask, other=0.0)
    
    # Fused multiply-add operation
    out = in2 * in1 + in0
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def fused_multiply_add(tmp_0, in_1, in_2):
    # Calculate the total number of elements
    # We need to handle broadcasting, so find the largest tensor size
    max_elements = max(tmp_0.numel(), in_1.numel(), in_2.numel())
    
    BLOCK_SIZE = 1024
    num_programs = (max_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Allocate output tensor with the same shape as the largest input
    if max_elements == tmp_0.numel():
        output_shape = tmp_0.shape
    elif max_elements == in_1.numel():
        output_shape = in_1.shape
    else:
        output_shape = in_2.shape
    
    output = torch.empty(output_shape, dtype=tmp_0.dtype, device=tmp_0.device)
    
    # Launch kernel
    fused_multiply_add_kernel[(num_programs,)](
        in0_ptr=tmp_0,
        in1_ptr=in_1,
        in2_ptr=in_2,
        out_ptr=output,
        n_elements=max_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_multiply_add