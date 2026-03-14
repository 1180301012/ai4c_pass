import torch
import triton
import triton.language as tl

def pattern(in_4, in_5):
    """Pattern for in_4 += in_5 element-wise addition"""
    in_4 += in_5
    return in_4

def replacement_args(in_4, in_5):
    """Extract arguments for the replacement kernel"""
    return (in_4, in_5)

@triton.jit
def elementwise_add_kernel(
    x_ptr,
    y_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """High-performance element-wise addition kernel"""
    # Each program handles a contiguous block of data
    block_start = tl.program_id(0) * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load inputs
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    
    # Compute and store result
    out = x + y
    tl.store(out_ptr + offsets, out, mask=mask)

@torch.fx.wrap
def elementwise_add(x, y):
    """Wrapper function launching the Triton kernel"""
    # Handle both tensor and scalar inputs safely
    if isinstance(x, (int, float)):
        # If first argument is scalar, treat as scalar addition
        if isinstance(y, torch.Tensor):
            return y + x
        else:
            return x + y
    
    n_elements = x.numel()
    
    # Dynamic BLOCK_SIZE selection based on tensor size
    if n_elements < 1024:
        BLOCK_SIZE = 256
    elif n_elements < 8192:
        BLOCK_SIZE = 512
    elif n_elements < 65536:
        BLOCK_SIZE = 1024
    else:
        BLOCK_SIZE = 2048
    
    num_programs = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Create output tensor
    out = torch.empty_like(x)
    
    # Use optimized kernel
    elementwise_add_kernel[(num_programs,)](
        x_ptr=x,
        y_ptr=y,
        out_ptr=out,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return out

def replacement_func():
    """Return the optimized function"""
    return elementwise_add