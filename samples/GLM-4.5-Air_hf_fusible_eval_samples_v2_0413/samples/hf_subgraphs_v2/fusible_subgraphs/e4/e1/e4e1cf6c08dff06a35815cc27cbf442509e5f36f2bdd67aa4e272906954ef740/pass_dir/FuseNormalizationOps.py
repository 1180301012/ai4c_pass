import torch
import triton
import triton.language as tl

def pattern(in_0, in_2):
    tmp_2 = in_0 * in_2
    return tmp_2

def replacement_args(in_0, in_2):
    return (in_0, in_2)

@triton.jit
def scaling_kernel(
    input_ptr,
    scalar_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate global indices for current program
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and multiply with scalar
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scalar = tl.load(scalar_ptr)
    result = x * scalar
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@triton.jit
def scaling_kernel_optimized(
    input_ptr,
    scalar_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    
    # Calculate global indices for current program
    block_offset = pid * BLOCK_SIZE
    offsets = block_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and multiply with scalar
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    scalar = tl.load(scalar_ptr)
    result = x * scalar
    
    # Store result
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_normalization(in_0, in_2):
    # Calculate total elements
    n_elements = in_0.numel()
    
    # Use larger block size for better GPU occupancy
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    output = torch.empty(in_0.shape, dtype=in_0.dtype, device=in_0.device)
    
    # Launch optimized kernel
    scaling_kernel_optimized[(num_programs,)](
        input_ptr=in_0,
        scalar_ptr=in_2,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output

def replacement_func():
    return fused_normalization