import torch
import triton
import triton.language as tl

def pattern(tmp_1, shape_0, shape_1):
    tmp_4 = tmp_1.new_zeros((shape_0, shape_1))
    return tmp_4

def replacement_args(tmp_1, shape_0, shape_1):
    return (tmp_1, shape_0, shape_1)

@triton.jit
def optimized_zeros_kernel(
    out_ptr,
    shape_0,
    shape_1,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program processes a block of elements
    program_id = tl.program_id(0)
    block_start = program_id * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Calculate 2D indices
    total_elements = shape_0 * shape_1
    mask = offsets < total_elements
    
    # Load zeros
    zeros = tl.zeros(1, dtype=tl.float32)  # This will be cast to appropriate dtype
    
    # Store zeros
    tl.store(out_ptr + offsets, zeros, mask=mask)

@torch.fx.wrap
def optimized_zeros(tmp_1, shape_0, shape_1):
    # Determine dtype from tmp_1
    dtype = tmp_1.dtype
    
    # Create output tensor
    out = torch.empty((shape_0, shape_1), dtype=dtype, device=tmp_1.device)
    
    # For small tensors, directly use torch.zeros
    if shape_0 * shape_1 <= 4096:
        out = torch.zeros((shape_0, shape_1), dtype=dtype, device=tmp_1.device)
    else:
        # Use Triton kernel for larger tensors
        BLOCK_SIZE = 1024
        total_elements = shape_0 * shape_1
        num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
        
        optimized_zeros_kernel[(num_programs,)](
            out_ptr=out,
            shape_0=shape_0,
            shape_1=shape_1,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return out

def replacement_func():
    return optimized_zeros