import torch
import triton
import triton.language as tl

def pattern(x, scale):
    """Simple pattern: element-wise multiplication x * scale"""
    return x * scale

def replacement_args(x, scale):
    """Extract arguments for the replacement function"""
    return (x, scale)

@triton.jit
def simple_multiply_kernel(
    x_ptr,
    scale_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple kernel for element-wise multiplication x * scale"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    scale = tl.load(scale_ptr + offsets, mask=mask, other=0.0)
    
    # Multiply
    result = x * scale
    
    # Store result
    tl.store(out_ptr + offsets, result, mask=mask)

@torch.fx.wrap
def simple_multiply_fused(x, scale):
    """Simple fused function: element-wise multiplication x * scale with broadcasting"""
    n_elements = x.numel()
    
    # Get tensor shapes for broadcasting
    x_shape = tuple(x.shape)
    scale_shape = tuple(scale.shape)
    
    # Output tensor
    result = torch.empty_like(x, dtype=torch.float32)
    
    # Optimal block size for this problem
    BLOCK_SIZE = 1024
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    simple_multiply_kernel[(num_programs,)](
        x_ptr=x,
        scale_ptr=scale,
        out_ptr=result,
        n_elements=n_elements,
        x_shape=x_shape,
        scale_shape=scale_shape,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return result

def replacement_func():
    """Return the fused function"""
    return simple_multiply_fused