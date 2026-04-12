import torch
import triton
import triton.language as tl

def pattern(tmp_1, shape):
    """Simple pattern for zeros creation"""
    tmp_4 = tmp_1.new_zeros(shape)
    return tmp_4

def replacement_args(tmp_1, shape):
    return (tmp_1, shape)

@triton.jit
def simple_zeros_kernel(
    out_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel for filling zeros"""
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, num_elements)
    
    # Fill with zeros
    for idx in range(start_idx, end_idx):
        tl.store(out_ptr + idx, 0.0)

@torch.fx.wrap
def simple_zeros_creation(source_tensor, shape):
    """Simple zeros creation using efficient Triton kernel"""
    M, D = shape
    num_elements = M * D
    
    # Use optimal block size
    if num_elements <= 1024:
        BLOCK_SIZE = num_elements
    else:
        BLOCK_SIZE = 256
    
    # Calculate grid size
    grid_size = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor
    out = torch.empty(shape, dtype=source_tensor.dtype, device=source_tensor.device)
    
    # Fill with zeros using optimized kernel
    simple_zeros_kernel[grid_size](
        out,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return simple_zeros_creation