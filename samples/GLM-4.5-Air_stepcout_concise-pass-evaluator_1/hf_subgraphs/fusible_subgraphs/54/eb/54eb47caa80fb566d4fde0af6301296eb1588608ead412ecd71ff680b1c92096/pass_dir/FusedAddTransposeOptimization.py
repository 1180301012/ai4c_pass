import torch
import triton
import triton.language as tl

def pattern(x, y):
    """Match basic addition operation"""
    return x + y

def replacement_args(x, y):
    """Extract arguments for the replacement kernel"""
    return (x, y)

@triton.jit
def simple_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple addition kernel that matches the pattern"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs with proper bounds checking
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Perform addition
    out = x + y
    
    # Store result
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def triton_add_optimized(x, y):
    """Optimized addition implementation with fallback"""
    # Only use Triton if tensors are simple and contiguous to avoid illegal memory access
    use_triton = (
        x.is_contiguous() and 
        y.is_contiguous() and 
        x.device == y.device and
        x.shape == y.shape and
        x.numel() > 1000  # Only for larger tensors where Triton helps
    )
    
    if use_triton:
        try:
            total_elements = x.numel()
            BLOCK_SIZE = 1024
            num_programs = (total_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
            
            out = torch.empty_like(x)
            simple_add_kernel[(num_programs,)](
                x_ptr=x,
                y_ptr=y,
                out_ptr=out,
                n_elements=total_elements,
                BLOCK_SIZE=BLOCK_SIZE,
            )
            return out
        except (RuntimeError, AssertionError):
            # Fall back to PyTorch addition
            pass
    
    # Use optimized PyTorch addition for most cases
    return x + y

def replacement_func():
    """Return the optimized function"""
    return triton_add_optimized