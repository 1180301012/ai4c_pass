import torch
import triton
import triton.language as tl

def pattern(x, min_val, max_val, inplace):
    """Pattern matches HardTanh operation"""
    return torch.nn.functional.hardtanh(x, min_val, max_val, inplace)

def replacement_args(x, min_val, max_val, inplace):
    """Extract arguments for the optimized kernel"""
    return (x, min_val, max_val, inplace)

@triton.jit
def hardtanh_kernel(
    input_ptr, output_ptr,
    n_elements, min_val, max_val,
    BLOCK_SIZE: tl.constexpr
):
    """Optimized HardTanh kernel using Triton"""
    pid = tl.program_id(0)
    start_idx = pid * BLOCK_SIZE
    end_idx = min(start_idx + BLOCK_SIZE, n_elements)
    
    if start_idx >= n_elements:
        return
    
    offsets = start_idx + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    
    # Apply HardTanh: clip between min_val and max_val
    y = tl.maximum(tl.minimum(x, max_val), min_val)
    
    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)

@torch.fx.wrap
def hardtanh_optimized(x, min_val, max_val, inplace):
    """Optimized HardTanh function"""
    # For this optimization, we ignore the inplace flag
    # Create output tensor
    output = torch.empty_like(x)
    
    # Apply optimized HardTanh kernel
    BLOCK_SIZE = 1024
    grid_size = (x.numel() + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    hardtanh_kernel[(grid_size,)](
        x, output,
        x.numel(), min_val, max_val, BLOCK_SIZE
    )
    
    return output

def replacement_func():
    """Returns the optimized function"""
    return hardtanh_optimized