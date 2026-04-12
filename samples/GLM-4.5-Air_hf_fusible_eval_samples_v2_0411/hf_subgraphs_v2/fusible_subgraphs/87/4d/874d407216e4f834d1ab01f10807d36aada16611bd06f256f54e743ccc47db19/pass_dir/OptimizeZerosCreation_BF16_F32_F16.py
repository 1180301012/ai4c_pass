import torch
import triton
import triton.language as tl

def pattern(tmp_1, shape):
    """Pattern: tmp_1.new_zeros((shape[0], shape[1]))"""
    tmp_4 = tmp_1.new_zeros(shape)
    return tmp_4

def replacement_args(tmp_1, shape):
    return (tmp_1, shape)

@triton.jit
def zeros_kernel(
    out_ptr,
    num_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Optimized kernel for creating zeros tensor using 1D threading"""
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, num_elements)
    
    # Fill contiguous block with zeros
    for idx in range(start_idx, end_idx):
        tl.store(out_ptr + idx, 0.0)

@torch.fx.wrap
def optimized_zeros_creation(source_tensor, shape):
    """Wrapper for optimized zeros tensor creation"""
    M, D = shape
    num_elements = M * D
    
    # Use optimal block size for zeros filling
    if num_elements <= 1024:
        BLOCK_SIZE = num_elements  # Single program for small tensors
    elif num_elements <= 16384:
        BLOCK_SIZE = 256  # Medium block size
    else:
        BLOCK_SIZE = 512  # Larger block size for big tensors
    
    # Calculate grid size (must be a tuple)
    grid_size = ((num_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    
    # Create output tensor
    out = torch.empty(shape, dtype=source_tensor.dtype, device=source_tensor.device)
    
    # Launch kernel
    zeros_kernel[grid_size](
        out,
        num_elements=num_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    return optimized_zeros_creation