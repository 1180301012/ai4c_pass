import torch
import triton
import triton.language as tl

def pattern(tmp_6, tmp_7):
    tmp_8 = tmp_6.masked_fill(tmp_7, -3.4028234663852886e+38)
    return tmp_8

def replacement_args(tmp_6, tmp_7):
    return (tmp_6, tmp_7)

# Autotuning configurations for optimized masked fill kernel
@triton.jit
def fused_masked_fill_kernel(
    input_ptr,
    mask_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # The program ID identifies the block of the program execution
    pid = tl.program_id(0)
    
    # Determine the memory offset for this program
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    
    # Load input values and mask values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    mask_vals = tl.load(mask_ptr + offsets, mask=mask, other=False)
    
    # Vectorized masked fill operation using tl.where
    # When mask_vals=True, use -infinity, otherwise use original value
    result = tl.where(mask_vals, -3.4028234663852886e+38, x)
    
    # Store the result back to global memory
    tl.store(output_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def fused_masked_fill(tmp_6, tmp_7):
    # Optimized masked fill: tmp_6.masked_fill(tmp_7, -inf)
    n_elements = tmp_6.numel()
    
    # Optimized block size for best GPU utilization
    BLOCK_SIZE = 1024
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor preserving dtype
    output = torch.empty_like(tmp_6)
    
    # Launch kernel with optimized configuration
    fused_masked_fill_kernel[(num_programs,)](
        input_ptr=tmp_6,
        mask_ptr=tmp_7,
        output_ptr=output,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output

def replacement_func():
    return fused_masked_fill